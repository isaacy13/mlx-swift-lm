// Copyright © 2026 Apple Inc.

import Dispatch
import Foundation
import MLX
import MLXLLM
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Regression test for the tool-call round trip: after dispatching tool calls,
/// the restarted generation must see BOTH the assistant message that made the
/// calls and the role:"tool" results. Chat templates that render tool results
/// by forward-scanning from an assistant message carrying `tool_calls`
/// (e.g. Gemma 4) otherwise drop the results from the rendered prompt.
public class ChatSessionToolRoundTripTests: XCTestCase {

    /// Ignores token ids and decodes a scripted string instead: the first
    /// generation pass yields a JSON tool call, later passes yield plain text.
    /// Stateful across passes, hence a class; calls are serialized by the
    /// session's generation task.
    private final class ScriptedToolCallTokenizer: MLXLMCommon.Tokenizer, @unchecked Sendable {
        private struct MissingUserQuery: Error {}

        private let requiresUserQuery: Bool
        private let toolCallPrefix: String
        private let plainOnly: Bool
        private let secondToolCity: String?
        private let decodeGate: DecodeGate?
        private let plainScript = "It is sunny in Paris today."
        private var pass = 0

        init(
            requiresUserQuery: Bool = true, toolCallPrefix: String = "",
            plainOnly: Bool = false, secondToolCity: String? = nil, decodeGate: DecodeGate? = nil
        ) {
            self.requiresUserQuery = requiresUserQuery
            self.toolCallPrefix = toolCallPrefix
            self.plainOnly = plainOnly
            self.secondToolCity = secondToolCity
            self.decodeGate = decodeGate
        }

        private var toolCallScript: String {
            let first = toolCallPrefix
                + #"<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>"#
            guard let secondToolCity else { return first }
            return first + "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"\(secondToolCity)\"}}</tool_call>"
        }

        let vocabularySize = 100
        private let _eosTokenId = 101
        private let _unknownTokenId = 102

        var bosToken: String? = nil
        var eosToken: String? = nil
        var eosTokenId: Int? { _eosTokenId }
        var unknownToken: String? = nil
        var unknownTokenId: Int? { _unknownTokenId }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            guard
                !requiresUserQuery
                    || messages.contains(where: { $0["role"] as? String == "user" })
            else {
                throw MissingUserQuery()
            }
            pass += 1
            return encode(text: "")
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            (0 ..< 8).map { _ in Int.random(in: 1 ..< vocabularySize) }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            decodeGate?.pauseSecondDecode()
            let script = !plainOnly && pass <= 1 ? toolCallScript : plainScript
            return String(script.prefix(min(tokenIds.count * 4, script.count)))
        }

        func convertTokenToId(_ token: String) -> Int? { 1 }
        func convertIdToToken(_ id: Int) -> String? { "x" }
    }

    private final class MessageLog: @unchecked Sendable {
        private let lock = NSLock()
        private var passes: [[Chat.Message]] = []

        func record(_ messages: [Chat.Message]) {
            lock.lock()
            passes.append(messages)
            lock.unlock()
        }

        var all: [[Chat.Message]] {
            lock.lock()
            defer { lock.unlock() }
            return passes
        }
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let log: MessageLog

        func generate(messages: [Chat.Message]) -> [Message] {
            log.record(messages)
            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    private final class DispatchLog: @unchecked Sendable {
        private let lock = NSLock()
        private var calls: [ToolCall] = []

        func record(_ call: ToolCall) {
            lock.lock()
            calls.append(call)
            lock.unlock()
        }

        var all: [ToolCall] {
            lock.lock()
            defer { lock.unlock() }
            return calls
        }
    }

    private static func makeContext(
        tokenizer: any Tokenizer, messageGenerator: MessageGenerator,
        customProcessor: (any UserInputProcessor)? = nil, headDimension: Int = 64
    ) -> ModelContext {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: headDimension,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test", toolCallFormat: .json),
            messageGenerator: messageGenerator)
        return .init(
            configuration: processor.configuration,
            model: model,
            processor: customProcessor ?? processor,
            tokenizer: tokenizer)
    }

    private static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": [
                "type": "object",
                "properties": [
                    "city": ["type": "string"] as [String: any Sendable]
                ] as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    func testRestartedGenerationSeesAssistantToolCallsBeforeResults() async throws {
        let log = MessageLog()
        let dispatched = DispatchLog()
        let tokenizer = ScriptedToolCallTokenizer()
        let context = Self.makeContext(
            tokenizer: tokenizer,
            messageGenerator: RecordingMessageGenerator(log: log))

        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 24),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return #"{"forecast": "sunny"}"#
            })

        let reply = try await session.respond(to: "What's the weather in Paris?")
        XCTAssertFalse(reply.isEmpty)

        // The scripted tool call was parsed and dispatched once.
        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(dispatched.all.first?.function.name, "get_weather")
        XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))

        // A restart happened, and its rendered transcript pairs the tool
        // result with the assistant message that made the call (the shape
        // asserted by ToolCallIdTests).
        let passes = log.all
        XCTAssertGreaterThanOrEqual(passes.count, 2)
        let restart = try XCTUnwrap(passes.last)
        XCTAssertTrue(
            restart.contains { $0.role == .user },
            "conversation-aware templates must retain the user query on tool continuation")

        let toolIndex = try XCTUnwrap(
            restart.lastIndex { $0.role == .tool },
            "restarted generation must include the tool result")
        XCTAssertEqual(restart[toolIndex].content, #"{"forecast": "sunny"}"#)

        guard toolIndex > 0 else {
            XCTFail("tool result was rendered with no preceding assistant tool_calls message")
            return
        }
        let assistant = restart[toolIndex - 1]
        XCTAssertEqual(assistant.role, .assistant)

        let rendered = DefaultMessageGenerator().generate(message: assistant)
        let toolCalls = try XCTUnwrap(
            rendered["tool_calls"] as? [[String: any Sendable]],
            "assistant message before the tool result must carry tool_calls")
        XCTAssertEqual(toolCalls.count, 1)
        let function = try XCTUnwrap(toolCalls.first?["function"] as? [String: any Sendable])
        XCTAssertEqual(function["name"] as? String, "get_weather")
    }

    func testGenerationPolicyControlsValidationBeforeAutomaticDispatch() async throws {
        let restrictedTool: ToolSpec = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "city": ["type": "string", "enum": ["Rome"]] as [String: any Sendable]
                    ],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        for validation in ToolCallValidationPolicy.allCases {
            let dispatched = DispatchLog()
            let context = Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(),
                messageGenerator: DefaultMessageGenerator())
            var policy = ToolCallPolicy(recovery: .disabled)
            if validation == .strict { policy.validation = .strict }
            let session = ChatSession(
                context,
                generateParameters: .init(maxTokens: 24, toolCallPolicy: policy),
                tools: [restrictedTool],
                toolDispatch: { call in
                    dispatched.record(call)
                    return "sunny"
                })
            do {
                _ = try await session.respond(to: "Weather in Paris?")
                XCTAssertEqual(validation, .permissive)
                XCTAssertEqual(dispatched.all.count, 1)
                XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))
            } catch let error as RejectedToolCallError {
                XCTAssertEqual(validation, .strict)
                XCTAssertEqual(error.rejection.reason, .invalidArguments)
                XCTAssertTrue(dispatched.all.isEmpty)
            }
        }
    }

    func testRawCacheToolRestartPreservesLegacyFragmentShape() async throws {
        let log = MessageLog()
        let tokenizer = ScriptedToolCallTokenizer(
            requiresUserQuery: false,
            toolCallPrefix: "Checking the tool. ")
        let context = Self.makeContext(
            tokenizer: tokenizer,
            messageGenerator: RecordingMessageGenerator(log: log))
        let parameters = GenerateParameters(maxTokens: 40)
        let rawCache = try context.model.newCache(parameters: parameters)
        let session = ChatSession(
            context,
            cache: rawCache,
            generateParameters: parameters,
            tools: [Self.weatherTool],
            toolDispatch: { _ in #"{"forecast": "sunny"}"# })

        _ = try await session.respond(to: "What's the weather in Paris?")

        let passes = log.all
        XCTAssertGreaterThanOrEqual(passes.count, 2)
        let restart = try XCTUnwrap(passes.last)
        XCTAssertFalse(
            restart.contains { $0.role == .user },
            "raw KV caches cannot reconstruct and replay their original transcript")

        let toolIndex = try XCTUnwrap(restart.lastIndex { $0.role == .tool })
        guard toolIndex > 0 else {
            XCTFail("tool result was rendered with no preceding assistant message")
            return
        }
        XCTAssertEqual(restart[toolIndex - 1].role, .assistant)
        XCTAssertEqual(
            restart[toolIndex - 1].content,
            "",
            "generated text is already represented by the raw cache and must not be replayed")
    }
    private actor PreparationGate {
        private var entered = false
        private var released = false
        private var entryWaiters: [CheckedContinuation<Void, Never>] = []
        private var releaseWaiter: CheckedContinuation<Void, Never>?

        func pauseOnce() async {
            guard !entered else { return }
            entered = true
            entryWaiters.forEach { $0.resume() }
            entryWaiters.removeAll()
            guard !released else { return }
            await withCheckedContinuation { releaseWaiter = $0 }
        }

        func waitUntilEntered() async {
            guard !entered else { return }
            await withCheckedContinuation { entryWaiters.append($0) }
        }

        var isPaused: Bool { entered && !released }

        func release() {
            released = true
            releaseWaiter?.resume()
            releaseWaiter = nil
        }
    }

    private struct PausedInputProcessor: UserInputProcessor {
        let base: TestInputProcessor
        let gate: PreparationGate

        func prepare(input: UserInput) async throws -> LMInput {
            await gate.pauseOnce()
            return try base.prepare(input: input)
        }
    }

    private actor PreparationFailure {
        private var shouldFail = true

        func consumeFailure() -> Bool {
            defer { shouldFail = false }
            return shouldFail
        }
    }

    private enum TestPreparationError: Error { case queuedWork }

    private struct FailingInputProcessor: UserInputProcessor {
        let base: TestInputProcessor
        let failure: PreparationFailure

        func prepare(input: UserInput) async throws -> LMInput {
            if await failure.consumeFailure() {
                let work = MLXArray.ones([64, 64]) + MLXArray.ones([64, 64])
                asyncEval(work)
                throw TestPreparationError.queuedWork
            }
            return try base.prepare(input: input)
        }
    }

    private final class DecodeGate: @unchecked Sendable {
        let entered: AsyncStream<Void>
        private let continuation: AsyncStream<Void>.Continuation
        private let lock = NSLock()
        private let releaseSignal = DispatchSemaphore(value: 0)
        private var count = 0

        init() {
            let pair = AsyncStream<Void>.makeStream()
            entered = pair.stream
            continuation = pair.continuation
        }

        func pauseSecondDecode() {
            lock.lock()
            count += 1
            let pause = count == 2
            lock.unlock()
            if pause {
                continuation.yield(())
                continuation.finish()
                releaseSignal.wait()
            }
        }

        func release() { releaseSignal.signal() }
    }

    private static func collect(
        _ response: (stream: AsyncThrowingStream<String, Error>, task: Task<Void, Never>)
    ) async throws -> String {
        var result = ""
        do {
            for try await chunk in response.stream { result += chunk }
        } catch {
            response.task.cancel()
            await response.task.value
            throw error
        }
        await response.task.value
        return result
    }

    func testProducerHandlePreservesActualToolRoundTrip() async throws {
        let log = MessageLog()
        let dispatched = DispatchLog()
        let session = ChatSession(
            Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(),
                messageGenerator: RecordingMessageGenerator(log: log)),
            generateParameters: .init(maxTokens: 24),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return #"{"forecast": "sunny"}"#
            })
        let reply = try await Self.collect(
            session.streamResponseTask(to: "What's the weather in Paris?"))
        XCTAssertEqual(reply, "It is sunny in Paris today.")
        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))
        let restart = try XCTUnwrap(log.all.last)
        let index = try XCTUnwrap(restart.lastIndex { $0.role == .tool })
        XCTAssertGreaterThan(index, 0)
        XCTAssertEqual(restart[index].content, #"{"forecast": "sunny"}"#)
        XCTAssertEqual(restart[index - 1].role, .assistant)
        let assistant = DefaultMessageGenerator().generate(message: restart[index - 1])
        let calls = try XCTUnwrap(assistant["tool_calls"] as? [[String: any Sendable]])
        XCTAssertEqual(calls.count, 1)
    }

    func testBatchProducerHandleKeepsCompletedHistory() async throws {
        let log = MessageLog()
        let session = ChatSession(
            Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(plainOnly: true),
                messageGenerator: RecordingMessageGenerator(log: log)),
            generateParameters: .init(maxTokens: 24))
        let first = try await Self.collect(
            session.streamResponseTask(to: [Chat.Message.user("first question")]))
        let second = try await Self.collect(
            session.streamResponseTask(to: [Chat.Message.user("second question")]))
        XCTAssertFalse(first.isEmpty)
        XCTAssertFalse(second.isEmpty)
        let messages = try XCTUnwrap(log.all.last)
        XCTAssertEqual(messages.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(messages[0].content, "first question")
        XCTAssertEqual(messages[1].content, first)
        XCTAssertEqual(messages[2].content, "second question")
    }

    func testCancelQueuedProducerBeforeCachePreparationDoesNotDispatch() async throws {
        let gate = PreparationGate()
        let log = MessageLog()
        let dispatched = DispatchLog()
        let tokenizer = ScriptedToolCallTokenizer()
        let base = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test", toolCallFormat: .json),
            messageGenerator: RecordingMessageGenerator(log: log))
        let session = ChatSession(
            Self.makeContext(
                tokenizer: tokenizer,
                messageGenerator: RecordingMessageGenerator(log: log),
                customProcessor: PausedInputProcessor(base: base, gate: gate)),
            generateParameters: .init(maxTokens: 24),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                return "sunny"
            })
        let firstPrompt = "first request holding the cache"
        let canceledPrompt = "queued request canceled before cache preparation"
        let first = session.streamResponseTask(to: firstPrompt)
        await gate.waitUntilEntered()
        let wasPaused = await gate.isPaused
        XCTAssertTrue(wasPaused)
        XCTAssertTrue(log.all.isEmpty)
        let canceled = session.streamResponseTask(to: canceledPrompt)
        canceled.task.cancel()
        XCTAssertTrue(canceled.task.isCancelled)
        let stillPaused = await gate.isPaused
        XCTAssertTrue(stillPaused)
        XCTAssertTrue(log.all.isEmpty)
        first.task.cancel()
        await gate.release()
        await first.task.value
        await canceled.task.value
        do {
            _ = try await Self.collect(first)
            XCTFail("Canceled cache owner unexpectedly succeeded")
        } catch is CancellationError {
        }
        do {
            _ = try await Self.collect(canceled)
            XCTFail("Canceled queued producer unexpectedly succeeded")
        } catch is CancellationError {
        }
        XCTAssertEqual(log.all.count, 1)
        XCTAssertEqual(log.all.first?.filter { $0.role == .user }.map(\.content), [firstPrompt])
        XCTAssertFalse(log.all.joined().contains { $0.content == canceledPrompt })
        XCTAssertTrue(dispatched.all.isEmpty)
        let recovered = try await Self.collect(session.streamResponseTask(to: "new question"))
        XCTAssertEqual(recovered, "It is sunny in Paris today.")
        XCTAssertTrue(dispatched.all.isEmpty)
        let finalMessages = try XCTUnwrap(log.all.last)
        XCTAssertEqual(finalMessages.filter { $0.role == .user }.map(\.content), ["new question"])
        XCTAssertFalse(finalMessages.contains { $0.content == firstPrompt || $0.content == canceledPrompt })
    }

    func testCancelDuringPreparationBeforeFirstYieldJoinsAndRecovers() async throws {
        let gate = PreparationGate()
        let log = MessageLog()
        let tokenizer = ScriptedToolCallTokenizer(plainOnly: true)
        let base = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test", toolCallFormat: .json),
            messageGenerator: RecordingMessageGenerator(log: log))
        let session = ChatSession(
            Self.makeContext(
                tokenizer: tokenizer,
                messageGenerator: RecordingMessageGenerator(log: log),
                customProcessor: PausedInputProcessor(base: base, gate: gate)),
            generateParameters: .init(maxTokens: 24))
        let response = session.streamResponseTask(to: "canceled during preparation")
        await gate.waitUntilEntered()
        response.task.cancel()
        await gate.release()
        await response.task.value
        do {
            _ = try await Self.collect(response)
            XCTFail("Canceled preparation unexpectedly yielded a successful response")
        } catch is CancellationError {
        }
        let recovered = try await Self.collect(session.streamResponseTask(to: "recovery question"))
        XCTAssertFalse(recovered.isEmpty)
        let last = try XCTUnwrap(log.all.last)
        XCTAssertEqual(last.filter { $0.role == .user }.map(\.content), ["recovery question"])
    }

    func testActiveDecodeCancelJoinsBeforeCacheRecovery() async throws {
        let gate = DecodeGate()
        let log = MessageLog()
        let session = ChatSession(
            Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(plainOnly: true, decodeGate: gate),
                messageGenerator: RecordingMessageGenerator(log: log)),
            generateParameters: .init(maxTokens: 128))
        let response = session.streamResponseTask(to: "interrupted question")
        var chunks = 0
        do {
            for try await chunk in response.stream {
                XCTAssertFalse(chunk.isEmpty)
                chunks += 1
                for await _ in gate.entered { break }
                response.task.cancel()
                gate.release()
                break
            }
        } catch {
            response.task.cancel()
            gate.release()
            await response.task.value
            throw error
        }
        gate.release()
        await response.task.value
        XCTAssertEqual(chunks, 1)
        session.generateParameters = .init(maxTokens: 1)
        let recovered = try await Self.collect(session.streamResponseTask(to: "fresh question"))
        XCTAssertFalse(recovered.isEmpty)
        let last = try XCTUnwrap(log.all.last)
        XCTAssertEqual(last.filter { $0.role == .user }.map(\.content), ["fresh question"])
    }

    func testCancelFirstDispatchPreventsSecondCallAndRestart() async throws {
        let gate = PreparationGate()
        let dispatched = DispatchLog()
        let log = MessageLog()
        let session = ChatSession(
            Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(secondToolCity: "Rome"),
                messageGenerator: RecordingMessageGenerator(log: log)),
            generateParameters: .init(maxTokens: 64),
            tools: [Self.weatherTool],
            toolDispatch: { call in
                dispatched.record(call)
                await gate.pauseOnce()
                return "sunny"
            })
        let response = session.streamResponseTask(to: "Weather in Paris and Rome?")
        await gate.waitUntilEntered()
        response.task.cancel()
        await gate.release()
        await response.task.value
        do {
            _ = try await Self.collect(response)
            XCTFail("Canceled dispatch unexpectedly succeeded")
        } catch is CancellationError {
        }
        XCTAssertEqual(dispatched.all.count, 1)
        XCTAssertEqual(dispatched.all.first?.function.arguments["city"], .string("Paris"))
        XCTAssertEqual(log.all.count, 1)
        let recovered = try await Self.collect(session.streamResponseTask(to: "fresh question"))
        XCTAssertFalse(recovered.isEmpty)
        XCTAssertEqual(dispatched.all.count, 1)
    }

    func testQueuedProcessorFailureThenValidRecovery() async throws {
        let log = MessageLog()
        let tokenizer = ScriptedToolCallTokenizer(plainOnly: true)
        let base = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "test", toolCallFormat: .json),
            messageGenerator: RecordingMessageGenerator(log: log))
        let session = ChatSession(
            Self.makeContext(
                tokenizer: tokenizer,
                messageGenerator: RecordingMessageGenerator(log: log),
                customProcessor: FailingInputProcessor(base: base, failure: PreparationFailure())),
            generateParameters: .init(maxTokens: 24))
        do {
            _ = try await Self.collect(session.streamResponseTask(to: "failed question"))
            XCTFail("Queued preparation failure was swallowed")
        } catch TestPreparationError.queuedWork {
        }
        let recovered = try await Self.collect(session.streamResponseTask(to: "recovered question"))
        XCTAssertEqual(recovered, "It is sunny in Paris today.")
        XCTAssertEqual(log.all.last?.filter { $0.role == .user }.map(\.content), ["recovered question"])
    }

    func testPlainModelContainerHandleAndLegacyGenerateOnCustomDefaultStream() async throws {
        try await Stream.withNewDefaultStream {
            let container = ModelContainer(context: Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(plainOnly: true),
                messageGenerator: DefaultMessageGenerator()))
            let input = try await container.prepare(input: UserInput(prompt: "plain generation"))
            let response = try await container.generateTask(
                input: input, parameters: .init(maxTokens: 24))
            var chunks = ""
            for await item in response.stream { chunks += item.chunk ?? "" }
            await response.task.value
            XCTAssertEqual(chunks, "It is sunny in Paris today.")
            let next = try await container.prepare(input: UserInput(prompt: "legacy generation"))
            let legacy = try await container.generate(input: next, parameters: .init(maxTokens: 24))
            var legacyChunks = ""
            for await item in legacy { legacyChunks += item.chunk ?? "" }
            XCTAssertEqual(legacyChunks, chunks)
        }
    }

    func testPostAsyncEvalIteratorValidationFailureThenPlainRecovery() async throws {
        try await Stream.withNewDefaultStream {
            let context = Self.makeContext(
                tokenizer: ScriptedToolCallTokenizer(plainOnly: true),
                messageGenerator: DefaultMessageGenerator(), headDimension: 48)
            let configuration = KVCacheConfiguration(
                strategy: .affine(try .init(bits: 4, groupSize: 64, compressionStart: 64)),
                compatibility: .requireAtLeastOneLayer)
            let parameters = GenerateParameters(maxTokens: 24, kvCache: configuration)
            let cache = try context.model.newCache(parameters: parameters)
            XCTAssertEqual(cache.count, 8)
            XCTAssertTrue(cache.allSatisfy { $0.offset == 0 })
            let initialGlobal = try XCTUnwrap(cache.compactMap { $0 as? KVCacheSimple }.first)
            XCTAssertTrue(initialGlobal.innerState().isEmpty)
            try validateKVCacheCompatibility(cache, configuration: configuration)
            do {
                let unexpected = try MLXLMCommon.generateTask(
                    input: LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7, 8])),
                    cache: cache, parameters: parameters, context: context)
                unexpected.task.cancel()
                await unexpected.task.value
                XCTFail("Unsupported realized cache dimensions unexpectedly succeeded")
            } catch let error as KVCacheConfigurationError {
                XCTAssertEqual(error, .noCompatibleLayers(strategy: .affine))
            }
            XCTAssertTrue(cache.allSatisfy { $0.offset == 8 })
            let realizedGlobal = try XCTUnwrap(cache.compactMap { $0 as? KVCacheSimple }.first)
            XCTAssertTrue(initialGlobal === realizedGlobal)
            let state = realizedGlobal.innerState()
            XCTAssertEqual(state.count, 2)
            XCTAssertEqual(state.first?.dim(3), 48)
            XCTAssertEqual(state.last?.dim(3), 48)
            do {
                try validateKVCacheCompatibility(cache, configuration: configuration)
                XCTFail("Realized unsupported cache unexpectedly passed independent validation")
            } catch let error as KVCacheConfigurationError {
                XCTAssertEqual(error, .noCompatibleLayers(strategy: .affine))
            }
            let response = try MLXLMCommon.generateTask(
                input: LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7, 8])),
                parameters: .init(maxTokens: 24), context: context)
            var chunks = ""
            for await item in response.stream { chunks += item.chunk ?? "" }
            await response.task.value
            XCTAssertEqual(chunks, "It is sunny in Paris today.")
        }
    }
}
