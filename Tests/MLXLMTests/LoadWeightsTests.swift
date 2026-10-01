// Copyright © 2026 Apple Inc.

import Cmlx
import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// A model with a head whose weights may live outside the selected weight files.
private class TwoLayerModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "layer") var layer: Linear
    @ModuleInfo(key: "projector") var projector: Linear

    override init() {
        _layer.wrappedValue = Linear(2, 2, bias: false)
        _projector.wrappedValue = Linear(2, 2, bias: false)
    }
}

/// The same model, declaring the sidecar its checkpoint ships the head in — like
/// `JinaRerankerModel` and `projector.safetensors`.
private final class SidecarDeclaringModel: TwoLayerModel, AdditionalWeightFilesProviding {
    var additionalWeightFiles: [String] { ["projector.safetensors"] }
}

private final class PreparedSidecarDeclaringModel: TwoLayerModel,
    AdditionalWeightFilesProviding, LanguageModel, KVCacheDimensionProvider
{
    var additionalWeightFiles: [String] { ["projector.safetensors"] }
    let kvHeads: [Int] = []
    private(set) var preparationCount = 0
    private(set) var projectorValuesAtPreparation: [Float] = []

    func prepare() throws {
        preparationCount += 1
        projectorValuesAtPreparation = projector.weight.asArray(Float.self)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        layer(inputs)
    }
}

private final class FailingInferenceStateModel: Module, LanguageModel,
    KVCacheDimensionProvider
{
    enum ExpectedFailure: Error { case preparation }

    let kvHeads: [Int] = []

    func prepare() throws {
        throw ExpectedFailure.preparation
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        inputs
    }
}

private func checkpointArrayIsAvailable(
    _ array: MLXArray, file: StaticString = #filePath, line: UInt = #line
) -> Bool {
    var available = false
    let status = _mlx_array_is_available(&available, array.ctx)
    XCTAssertEqual(status, 0, "Pinned internal array availability check failed", file: file, line: line)
    return available
}

private final class CheckpointSanitizationProbe {
    var invocationCount = 0
    var inputKeys: Set<String> = []
    var metadata: [String: String] = [:]
    var discardedTensor: MLXArray?
    var discardedWasAvailableAtEntry: Bool?
    var inspectedLayerValues: [Float] = []
    var preparationCount = 0
    var layerValuesAtPreparation: [Float] = []
    var projectorValuesAtPreparation: [Float] = []

    func record(weights: [String: MLXArray], metadata: [String: String]) {
        invocationCount += 1
        inputKeys = Set(weights.keys)
        self.metadata = metadata
        discardedTensor = weights["unused.weight"]
        if let discardedTensor {
            discardedWasAvailableAtEntry = checkpointArrayIsAvailable(discardedTensor)
        }
    }
}

private final class SanitizingCheckpointModel: Module,
    AdditionalWeightFilesProviding, LanguageModel, KVCacheDimensionProvider
{
    @ModuleInfo(key: "layer") var layer: Linear
    @ModuleInfo(key: "projector") var projector: Linear
    let probe = CheckpointSanitizationProbe()
    let declaresSidecar: Bool
    let kvHeads: [Int] = []
    var additionalWeightFiles: [String] { declaresSidecar ? ["projector.safetensors"] : [] }

    init(declaresSidecar: Bool = true) {
        self.declaresSidecar = declaresSidecar
        _layer.wrappedValue = Linear(2, 2, bias: false)
        _projector.wrappedValue = Linear(2, 2, bias: false)
        super.init()
    }

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        probe.record(weights: weights, metadata: metadata)
        var result: [String: MLXArray] = [:]
        if let source = weights["checkpoint.layer"] {
            probe.inspectedLayerValues = source.asArray(Float.self)
            let retained = source[0..<2, 0..<2]
            result["layer.weight"] = metadata["transpose"] == "true" ? retained.T : retained
        }
        if let source = weights["checkpoint.projector"] {
            result["projector.weight"] = source
        }
        return result
    }

    func prepare() throws {
        probe.preparationCount += 1
        probe.layerValuesAtPreparation = layer.weight.asArray(Float.self)
        probe.projectorValuesAtPreparation = projector.weight.asArray(Float.self)
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        layer(inputs)
    }
}

private final class SanitizingQuantizedCheckpointModel: Module, BaseLanguageModel,
    AdditionalWeightFilesProviding
{
    @ModuleInfo(key: "layer") var layer: Linear
    @ModuleInfo(key: "projector") var projector: Linear
    let probe = CheckpointSanitizationProbe()
    var additionalWeightFiles: [String] { ["projector.safetensors"] }

    override init() {
        _layer.wrappedValue = Linear(32, 2, bias: false)
        _projector.wrappedValue = Linear(32, 2, bias: false)
        super.init()
    }

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        probe.record(weights: weights, metadata: metadata)
        var result: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix("checkpoint.") {
            result[String(key.dropFirst("checkpoint.".count))] = value
        }
        return result
    }
}
final class LoadWeightsTests: XCTestCase {

    // MARK: - Concurrent loading

    func testContiguousLoadGroupsBalanceBytesAndPreserveOrder() {
        // one huge tensor between small ones: boundaries land after the bytes, never inside
        let groups = contiguousLoadGroups(byteCounts: [1, 1, 100, 1, 1], groupCount: 2)
        XCTAssertEqual(groups, [0 ..< 3, 3 ..< 5])

        let even = contiguousLoadGroups(byteCounts: [10, 10, 10, 10], groupCount: 2)
        XCTAssertEqual(even, [0 ..< 2, 2 ..< 4])

        // every index appears exactly once, in order
        let many = contiguousLoadGroups(byteCounts: Array(repeating: 7, count: 100), groupCount: 16)
        XCTAssertEqual(many.flatMap { Array($0) }, Array(0 ..< 100))
    }

    func testContiguousLoadGroupsDegenerateInputs() {
        XCTAssertEqual(contiguousLoadGroups(byteCounts: [], groupCount: 4), [])
        XCTAssertEqual(contiguousLoadGroups(byteCounts: [5], groupCount: 4), [0 ..< 1])
        XCTAssertEqual(contiguousLoadGroups(byteCounts: [0, 0], groupCount: 4), [0 ..< 2])
        XCTAssertEqual(contiguousLoadGroups(byteCounts: [1, 2, 3], groupCount: 1), [0 ..< 3])
    }

    func testSafetensorSpansComeBackInFileOrder() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("model.safetensors")
        try save(
            arrays: [
                "a": MLXArray.zeros([4, 4]),
                "b": MLXArray.zeros([2]),
                "c": MLXArray.zeros([8, 8]),
            ], url: url)

        let spans = try safetensorSpansInFileOrder(url: url)

        XCTAssertEqual(Set(spans.map(\.name)), ["a", "b", "c"])
        XCTAssertEqual(
            spans.first { $0.name == "a" }?.byteCount, 4 * 4 * 4, "float32 4x4")
        // the order is the file's own layout, whatever it is, and covers each tensor once
        XCTAssertEqual(spans.count, 3)
    }

    func testSafetensorSpansRejectsAFileThatIsNotSafetensors() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("weights.safetensors")
        try Data("not a safetensors file at all".utf8).write(to: url)

        XCTAssertThrowsError(try safetensorSpansInFileOrder(url: url))
    }

    func testLoadWeightArraysMatchesTheSerialLoader() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // two shards with enough tensors to split, plus a duplicate name whose
        // later-file-wins resolution must match the serial loop
        var first = [String: MLXArray]()
        for i in 0 ..< 8 {
            first["layers.\(i).weight"] = MLXArray(Float(i)) * MLXArray.ones([16, 16])
        }
        first["shared.weight"] = MLXArray.zeros([4])
        var second = [String: MLXArray]()
        for i in 8 ..< 12 {
            second["layers.\(i).weight"] = MLXArray(Float(i)) * MLXArray.ones([16, 16])
        }
        second["shared.weight"] = MLXArray.ones([4])

        let urls = [
            directory.appendingPathComponent("model-00001-of-00002.safetensors"),
            directory.appendingPathComponent("model-00002-of-00002.safetensors"),
        ]
        try save(arrays: first, url: urls[0])
        try save(arrays: second, url: urls[1])

        let (weights, _) = try loadWeightArrays(urls: urls)

        var serial = [String: MLXArray]()
        for url in urls {
            let (w, _) = try loadArraysAndMetadata(url: url)
            serial.merge(w) { _, new in new }
        }

        XCTAssertEqual(Set(weights.keys), Set(serial.keys))
        for (name, expected) in serial {
            let actual = try XCTUnwrap(weights[name])
            XCTAssertEqual(actual.shape, expected.shape, name)
            XCTAssertTrue(
                allClose(actual, expected).item(Bool.self), "\(name) differs from serial load")
        }
        // the duplicate resolves to the later file, as the serial loop does
        XCTAssertEqual(weights["shared.weight"]?.asArray(Float.self), [1, 1, 1, 1])
    }

    func testLoadWeightArraysSurfacesAMissingFile() {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-missing-\(UUID().uuidString).safetensors")
        XCTAssertThrowsError(try loadWeightArrays(urls: [missing]))
    }

    func testWeightLoadConcurrencyIsClamped() {
        XCTAssertEqual(weightLoadConcurrency(processorCount: 2), 4)
        XCTAssertEqual(weightLoadConcurrency(processorCount: 8), 8)
        XCTAssertEqual(weightLoadConcurrency(processorCount: 14), 14)
        XCTAssertEqual(weightLoadConcurrency(processorCount: 32), 16)
    }

    func testInferencePreparationFailureIsReportedWithoutEscaping() {
        let report = prepareInferenceState(in: FailingInferenceStateModel())

        XCTAssertFalse(report.succeeded)
        XCTAssertEqual(report.failures.count, 1)
        XCTAssertTrue(report.failures[0].modelType.contains("FailingInferenceStateModel"))
        XCTAssertTrue(
            report.failures[0].error is FailingInferenceStateModel.ExpectedFailure)
    }

    // MARK: - Index

    func testIndexSelectsOnlyTheFilesItNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("model-extra.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testIndexMayNameFilesInSubdirectories() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // An index naming a nested file is a deliberate statement about where this model's
        // weights live, unlike a nested file nobody claims.
        try writeEmptyFile("shards/model-00001-of-00001.safetensors", in: directory)
        try writeIndex(
            ["model.norm.weight": "shards/model-00001-of-00001.safetensors"], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model-00001-of-00001.safetensors"])
    }

    // MARK: - Convention fallback

    func testStaleIndexFallsBackToTheConventionalNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // mlx-community/Qwen3-VL-4B-Instruct-4bit: one `model.safetensors`, but an index
        // carried over from the unquantized source repo that names two shards it never shipped.
        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("head.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            ], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        // the convention picks the weights back up without dragging in an unrelated file
        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testPartiallyStaleIndexFallsBackToTheConventionalNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            ], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model-00001-of-00002.safetensors"])
    }

    func testNoIndexUsesTheConventionalNames() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeEmptyFile("model-00002-of-00002.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(
            names, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    }

    func testFallsBackToWeightNamesThenToEverythingPresent() throws {
        let weightPrefixed = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: weightPrefixed) }
        try writeEmptyFile("weights.safetensors", in: weightPrefixed)
        try writeEmptyFile("prerotated_cache.safetensors", in: weightPrefixed)

        XCTAssertEqual(
            try safetensorWeightURLs(in: weightPrefixed).map(\.lastPathComponent),
            ["weights.safetensors"])

        let unconventional = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: unconventional) }
        try writeEmptyFile("adapters.safetensors", in: unconventional)

        // nothing conventional to go on: load what is there rather than nothing at all
        XCTAssertEqual(
            try safetensorWeightURLs(in: unconventional).map(\.lastPathComponent),
            ["adapters.safetensors"])
    }

    // MARK: - Subdirectories

    func testNestedWeightFilesAreNeverSelectedOnTheirOwn() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // mlx-community/Qwen3.5-4B-OptiQ-4bit ships its auxiliary weights under `optiq/`.
        // Loading those into the model corrupts generation silently (#408), so they must stay
        // out however the weight files are chosen -- including a nested HF snapshot cache under
        // a local checkpoint directory.
        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("optiq/mtp.safetensors", in: directory)
        try writeEmptyFile("optiq/optiq_vision.safetensors", in: directory)

        for selection in [WeightFileSelection.automatic, .allFilesPresent] {
            XCTAssertEqual(
                try safetensorWeightURLs(in: directory, selection: selection)
                    .map(\.lastPathComponent),
                ["model.safetensors"],
                "\(selection) must not descend into subdirectories")
        }

        // ... and the same with an index that no longer matches the shipped files
        try writeIndex(
            ["model.norm.weight": "model-00001-of-00002.safetensors"], in: directory)
        XCTAssertEqual(
            try safetensorWeightURLs(in: directory).map(\.lastPathComponent),
            ["model.safetensors"])
    }

    // MARK: - Additional files

    func testAdditionalFilesAreAppendedAndDeduplicated() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("projector.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        // the selected file comes first so its metadata wins
        XCTAssertEqual(
            try safetensorWeightURLs(
                in: directory,
                additionalFiles: ["projector.safetensors", "missing.safetensors"]
            ).map(\.lastPathComponent),
            ["model.safetensors", "projector.safetensors"])

        // an already-selected file is not loaded twice
        XCTAssertEqual(
            try safetensorWeightURLs(
                in: directory, additionalFiles: ["model.safetensors"]
            ).map(\.lastPathComponent),
            ["model.safetensors"])
    }

    // MARK: - Caller policy

    func testAllFilesPresentOverridesTheIndex() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("head.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        XCTAssertEqual(
            try safetensorWeightURLs(in: directory, selection: .allFilesPresent)
                .map(\.lastPathComponent),
            ["head.safetensors", "model.safetensors"])
    }

    // MARK: - loadWeights end to end

    func testLoadWeightsReadsSidecarWeightsDeclaredByTheModel() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        let model = SidecarDeclaringModel()
        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.projector.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testLoadWeightsPreparesInferenceStateAfterInstallingParameters() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        let model = PreparedSidecarDeclaringModel()
        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.preparationCount, 1)
        XCTAssertEqual(model.projectorValuesAtPreparation, [1, 2, 3, 4])
    }

    func testLoadWeightsFailsWhenTheSidecarIsNotDeclared() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // Neither the index nor the `model*` convention covers `projector.safetensors`, so the
        // head is never loaded -- `verify: [.all]` is what turns that into the keyNotFound
        // error from #560 instead of a silently untrained head.
        let model = TwoLayerModel()
        XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model))
    }

    func testLoadWeightsHonorsTheCallerSelectionPolicy() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // The escape hatch for a checkpoint no model in the registry knows about.
        let model = TwoLayerModel()
        try loadWeights(
            modelDirectory: directory, model: model, weightFileSelection: .allFilesPresent)

        XCTAssertEqual(model.projector.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testAsyncLoadWeightsMatchesTheSynchronousOverload() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // In an async context this resolves to the async overload, which suspends the caller
        // and runs the load on a global queue instead of blocking a cooperative thread.
        let model = SidecarDeclaringModel()
        try await loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.projector.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testAsyncLoadWeightsPropagatesErrors() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // same undeclared-sidecar failure as the synchronous overload
        let model = TwoLayerModel()
        do {
            try await loadWeights(modelDirectory: directory, model: model)
            XCTFail("expected the keyNotFound error the synchronous overload throws")
        } catch {
            // expected
        }
    }

    func testPinnedAvailabilityObserverDoesNotEvaluateLazyArray() {
        let lazy = MLXArray(converting: [1.0, 2, 3, 4]) + 1
        for _ in 0..<3 {
            XCTAssertFalse(checkpointArrayIsAvailable(lazy))
        }
        eval(lazy)
        XCTAssertTrue(checkpointArrayIsAvailable(lazy))
        XCTAssertEqual(lazy.asArray(Float.self), [2, 3, 4, 5])
        XCTAssertTrue(checkpointArrayIsAvailable(lazy))
    }

    func testLoadWeightsSanitizesCompleteCheckpointBeforeEvaluatingDiscardedTensor() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeSanitizingCheckpoint(in: directory, metadata: ["format": "fixture", "transpose": "true"])
        let model = SanitizingCheckpointModel()

        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.probe.invocationCount, 1)
        XCTAssertEqual(model.probe.inputKeys, ["checkpoint.layer", "checkpoint.projector", "unused.weight"])
        XCTAssertEqual(model.probe.metadata, ["format": "fixture", "transpose": "true"])
        XCTAssertEqual(model.probe.discardedWasAvailableAtEntry, false)
        let discarded = try XCTUnwrap(model.probe.discardedTensor)
        XCTAssertEqual(discarded.dtype, .float32)
        XCTAssertFalse(checkpointArrayIsAvailable(discarded))
        XCTAssertEqual(model.probe.inspectedLayerValues, [1, 2, 3, 4, 90, 91])
        XCTAssertEqual(model.layer.weight.shape, [2, 2])
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 3, 2, 4])
        XCTAssertEqual(model.projector.weight.asArray(Float.self), [31, 32, 33, 34])
        XCTAssertEqual(model.probe.preparationCount, 1)
        XCTAssertEqual(model.probe.layerValuesAtPreparation, [1, 3, 2, 4])
        XCTAssertEqual(model.probe.projectorValuesAtPreparation, [31, 32, 33, 34])
        XCTAssertFalse(checkpointArrayIsAvailable(discarded))
    }

    func testLoadWeightArraysDefaultStillMaterializesEverySelectedTensor() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeSanitizingCheckpoint(in: directory)
        let urls = ["model.safetensors", "projector.safetensors"].map {
            directory.appendingPathComponent($0)
        }

        let (weights, _) = try loadWeightArrays(urls: urls)

        XCTAssertEqual(Set(weights.keys), ["checkpoint.layer", "checkpoint.projector", "unused.weight"])
        for key in ["checkpoint.layer", "checkpoint.projector", "unused.weight"] {
            let array = try XCTUnwrap(weights[key])
            XCTAssertEqual(array.dtype, .float32)
            XCTAssertTrue(checkpointArrayIsAvailable(array), key)
        }
    }

    func testLoadWeightsSanitizesMergedShardsWithLaterDuplicateAndFirstMetadata() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try save(
            arrays: [
                "checkpoint.layer": MLXArray(converting: [1.0, 2, 3, 4, 90, 91]).reshaped(3, 2),
                "unused.weight": MLXArray(converting: [100.0, 101, 102, 103]),
            ],
            metadata: ["transpose": "true", "source": "first"],
            url: directory.appendingPathComponent("model-00001-of-00002.safetensors"))
        try save(
            arrays: [
                "checkpoint.layer": MLXArray(converting: [11.0, 12, 13, 14, 90, 91]).reshaped(3, 2),
                "checkpoint.projector": MLXArray.zeros([2, 2]),
            ],
            metadata: ["transpose": "false", "source": "second"],
            url: directory.appendingPathComponent("model-00002-of-00002.safetensors"))
        try save(
            arrays: ["checkpoint.projector": MLXArray(converting: [31.0, 32, 33, 34]).reshaped(2, 2)],
            metadata: ["transpose": "false", "source": "sidecar"],
            url: directory.appendingPathComponent("projector.safetensors"))
        try writeIndex([
            "checkpoint.layer": "model-00001-of-00002.safetensors",
            "checkpoint.projector": "model-00002-of-00002.safetensors",
        ], in: directory)
        let model = SanitizingCheckpointModel()

        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.probe.invocationCount, 1)
        XCTAssertEqual(model.probe.inputKeys, ["checkpoint.layer", "checkpoint.projector", "unused.weight"])
        XCTAssertEqual(model.probe.metadata, ["transpose": "true", "source": "first"])
        XCTAssertEqual(model.probe.inspectedLayerValues, [11, 12, 13, 14, 90, 91])
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [11, 13, 12, 14])
        XCTAssertEqual(model.projector.weight.asArray(Float.self), [31, 32, 33, 34])
        XCTAssertEqual(model.probe.preparationCount, 1)
        XCTAssertEqual(model.probe.layerValuesAtPreparation, [11, 13, 12, 14])
        XCTAssertEqual(model.probe.projectorValuesAtPreparation, [31, 32, 33, 34])
    }

    func testLoadWeightsUsesFirstNonemptyMetadataWhenFirstSelectedFileHasNone() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let expectedMetadata = ["transpose": "true", "source": "sidecar"]
        try writeSanitizingCheckpoint(in: directory, metadata: [:], sidecarMetadata: expectedMetadata)
        let model = SanitizingCheckpointModel()

        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.probe.invocationCount, 1)
        XCTAssertEqual(model.probe.inputKeys, ["checkpoint.layer", "checkpoint.projector", "unused.weight"])
        XCTAssertEqual(model.probe.metadata, expectedMetadata)
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 3, 2, 4])
        XCTAssertEqual(model.probe.layerValuesAtPreparation, [1, 3, 2, 4])
        XCTAssertEqual(model.probe.preparationCount, 1)
    }

    func testLoadWeightsQuantizesSanitizedRetainedLayerWithoutQuantizingDenseSidecar() throws {
        try assertSanitizedQuantizedCheckpoint(perLayerQuantization: false)
    }

    func testLoadWeightsUsesPerLayerQuantizationAfterCheckpointRemapping() throws {
        try assertSanitizedQuantizedCheckpoint(perLayerQuantization: true)
    }

    func testLoadWeightsMissingDeclaredParametersFailsBeforeInferencePreparation() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeSanitizingCheckpoint(in: directory)
        let model = SanitizingCheckpointModel(declaresSidecar: false)

        XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model))

        XCTAssertEqual(model.probe.invocationCount, 1)
        XCTAssertEqual(model.probe.inputKeys, ["checkpoint.layer", "unused.weight"])
        XCTAssertEqual(model.probe.preparationCount, 0)
        XCTAssertEqual(model.probe.layerValuesAtPreparation, [])
        XCTAssertEqual(model.probe.projectorValuesAtPreparation, [])
    }

    func testLoadWeightsMalformedSelectedFileFailsBeforeSanitizing() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("not a safetensors file".utf8).write(
            to: directory.appendingPathComponent("model.safetensors"))
        let model = SanitizingCheckpointModel()

        XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model))

        XCTAssertEqual(model.probe.invocationCount, 0)
        XCTAssertEqual(model.probe.preparationCount, 0)
        XCTAssertNil(model.probe.discardedTensor)
    }

    func testAsyncLoadWeightsAlsoSanitizesBeforeDiscardedTensorMaterialization() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeSanitizingCheckpoint(in: directory, metadata: ["transpose": "true"])
        let model = SanitizingCheckpointModel()

        try await loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.probe.invocationCount, 1)
        XCTAssertEqual(model.probe.inputKeys, ["checkpoint.layer", "checkpoint.projector", "unused.weight"])
        XCTAssertEqual(model.probe.metadata, ["transpose": "true"])
        XCTAssertEqual(model.probe.discardedWasAvailableAtEntry, false)
        let discarded = try XCTUnwrap(model.probe.discardedTensor)
        XCTAssertFalse(checkpointArrayIsAvailable(discarded))
        XCTAssertEqual(model.probe.inspectedLayerValues, [1, 2, 3, 4, 90, 91])
        XCTAssertEqual(model.layer.weight.asArray(Float.self), [1, 3, 2, 4])
        XCTAssertEqual(model.projector.weight.asArray(Float.self), [31, 32, 33, 34])
        XCTAssertEqual(model.probe.preparationCount, 1)
        XCTAssertEqual(model.probe.layerValuesAtPreparation, [1, 3, 2, 4])
        XCTAssertEqual(model.probe.projectorValuesAtPreparation, [31, 32, 33, 34])
    }

    private func writeSanitizingCheckpoint(
        in directory: URL, metadata: [String: String] = [:],
        sidecarMetadata: [String: String] = [:]
    ) throws {
        try save(
            arrays: [
                "checkpoint.layer": MLXArray(converting: [1.0, 2, 3, 4, 90, 91]).reshaped(3, 2),
                "unused.weight": MLXArray(converting: [100.0, 101, 102, 103]),
            ],
            metadata: metadata, url: directory.appendingPathComponent("model.safetensors"))
        try save(
            arrays: ["checkpoint.projector": MLXArray(converting: [31.0, 32, 33, 34]).reshaped(2, 2)],
            metadata: sidecarMetadata, url: directory.appendingPathComponent("projector.safetensors"))
        try writeIndex(["checkpoint.layer": "model.safetensors"], in: directory)
    }

    private func assertSanitizedQuantizedCheckpoint(perLayerQuantization: Bool) throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = MLXArray((0..<64).map { Float($0 % 32) }).reshaped(2, 32)
        let packed = MLX.quantized(source, groupSize: 32, bits: 4)
        let biases = try XCTUnwrap(packed.biases)
        let projector = MLXArray((0..<64).map { Float($0) }).reshaped(2, 32)
        try save(
            arrays: [
                "checkpoint.layer.weight": packed.wq,
                "checkpoint.layer.scales": packed.scales,
                "checkpoint.layer.biases": biases,
                "unused.weight": MLXArray(converting: [100.0, 101]),
            ],
            metadata: ["format": "quantized-fixture"],
            url: directory.appendingPathComponent("model.safetensors"))
        try save(
            arrays: ["checkpoint.projector.weight": projector],
            url: directory.appendingPathComponent("projector.safetensors"))
        try writeIndex(["checkpoint.layer.weight": "model.safetensors"], in: directory)
        let model = SanitizingQuantizedCheckpointModel()
        if perLayerQuantization {
            try loadWeights(
                modelDirectory: directory, model: model,
                perLayerQuantization: BaseConfiguration.PerLayerQuantization(
                    quantization: .init(groupSize: 64, bits: 8),
                    perLayerQuantization: [
                        "layer": .quantize(.init(groupSize: 32, bits: 4)),
                        "projector": .skip,
                    ]))
        } else {
            try loadWeights(
                modelDirectory: directory, model: model,
                quantization: .init(groupSize: 32, bits: 4))
        }

        XCTAssertEqual(model.probe.invocationCount, 1)
        XCTAssertEqual(model.probe.inputKeys, [
            "checkpoint.layer.weight", "checkpoint.layer.scales", "checkpoint.layer.biases",
            "checkpoint.projector.weight", "unused.weight",
        ])
        XCTAssertEqual(model.probe.metadata, ["format": "quantized-fixture"])
        let layer = try XCTUnwrap(model.layer as? QuantizedLinear)
        XCTAssertEqual(layer.groupSize, 32)
        XCTAssertEqual(layer.bits, 4)
        XCTAssertEqual(layer.weight.asArray(UInt32.self), packed.wq.asArray(UInt32.self))
        XCTAssertEqual(layer.scales.asArray(Float.self), packed.scales.asArray(Float.self))
        XCTAssertEqual(try XCTUnwrap(layer.biases).asArray(Float.self), biases.asArray(Float.self))
        XCTAssertFalse(model.projector is QuantizedLinear)
        XCTAssertEqual(model.projector.weight.asArray(Float.self), (0..<64).map { Float($0) })
        let input = MLXArray.ones([1, 32])
        let expected = MLX.quantizedMM(
            input, packed.wq, scales: packed.scales, biases: biases,
            transpose: true, groupSize: 32, bits: 4)
        XCTAssertEqual(model.layer(input).asArray(Float.self), expected.asArray(Float.self))
    }

    /// Writes a checkpoint whose index names only `model.safetensors` while the head lives in
    /// `projector.safetensors`, the `jinaai/jina-reranker-v3-mlx` layout.
    private func writeSidecarCheckpoint(in directory: URL) throws {
        try save(
            arrays: ["layer.weight": MLXArray.zeros([2, 2])],
            url: directory.appendingPathComponent("model.safetensors"))
        try save(
            arrays: [
                "projector.weight": MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(2, 2)
            ],
            url: directory.appendingPathComponent("projector.safetensors"))
        try writeIndex(["layer.weight": "model.safetensors"], in: directory)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        let url = directory.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data().write(to: url)
    }

    private func writeIndex(_ weightMap: [String: String], in directory: URL) throws {
        let index: [String: Any] = [
            "metadata": ["total_size": 1],
            "weight_map": weightMap,
        ]
        let data = try JSONSerialization.data(withJSONObject: index)
        try data.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }
}
