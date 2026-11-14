"""Offline checks for ClaudeController and DenseRetriever with fake clients. No API calls."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from agentic import ClaudeController, DenseRetriever, run_agent  # noqa: E402


def tool_use(name, inp):
    return types.SimpleNamespace(type='tool_use', name=name, input=inp)


def response(blocks):
    return types.SimpleNamespace(content=blocks, usage=types.SimpleNamespace(input_tokens=10, output_tokens=5))


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


CHUNK = {'id': 'c1', 'text': 'Hoàn tiền trong 7 ngày.', 'source_id': 'doc', 'source': 'doc.pdf', 'page': 1, 'metadata': {}}


class ClaudeControllerTests(unittest.TestCase):
    def test_search_action_is_normalized(self):
        client = FakeClient([response([tool_use('search', {'query': 'hoàn tiền'})])])
        action = ClaudeController('m', client=client).decide('Hoàn tiền?', [])
        self.assertEqual(action, {'action': 'search', 'query': 'hoàn tiền'})

    def test_answer_action_keeps_citations(self):
        client = FakeClient([response([tool_use('answer', {'answer': '7 ngày [c1]', 'citations': ['c1']})])])
        action = ClaudeController('m', client=client).decide('q', [])
        self.assertEqual(action, {'action': 'answer', 'answer': '7 ngày [c1]', 'citations': ['c1']})

    def test_takes_first_tool_use_and_counts_usage(self):
        client = FakeClient([response([tool_use('search', {'query': 'a'}), tool_use('answer', {'answer': 'x', 'citations': []})])])
        controller = ClaudeController('m', client=client)
        self.assertEqual(controller.decide('q', [])['action'], 'search')
        self.assertEqual(controller.usage, {'calls': 1, 'input_tokens': 10, 'output_tokens': 5})

    def test_no_tool_use_raises(self):
        client = FakeClient([response([types.SimpleNamespace(type='text', text='xin chào')])])
        with self.assertRaises(ValueError):
            ClaudeController('m', client=client).decide('q', [])

    def test_request_forces_a_tool_and_passes_context(self):
        client = FakeClient([response([tool_use('search', {'query': 'b'})])])
        history = [{'queries': ['a'], 'context': '[c1 | doc.pdf | page 1]\nHoàn tiền trong 7 ngày.'}]
        ClaudeController('m', client=client).decide('q', history)
        call = client.messages.calls[0]
        self.assertEqual(call['tool_choice'], {'type': 'any'})
        self.assertNotIn('temperature', call)
        self.assertIn('Hoàn tiền trong 7 ngày.', call['messages'][0]['content'])

    def test_full_loop_with_fake_model_and_store(self):
        client = FakeClient([response([tool_use('search', {'query': 'hoàn tiền'})]),
                             response([tool_use('answer', {'answer': '7 ngày [c1]', 'citations': ['c1']})])])

        class Store:
            def search(self, strategy, vector, top_k=3, filters=None):
                return [{'id': 'c1', 'score': 0.9}]

        class Embedder:
            dimensions = 3

            def embed_query(self, q):
                return [0.1, 0.2, 0.3]

        retriever = DenseRetriever(Store(), Embedder(), [CHUNK])
        result = run_agent('Hoàn tiền?', ClaudeController('m', client=client), retriever, budget=1200)
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['c1'])


class DenseRetrieverTests(unittest.TestCase):
    def test_rows_become_day7_chunks_with_score(self):
        class Store:
            def search(self, strategy, vector, top_k=3, filters=None):
                self.args = (strategy, top_k, filters)
                return [{'id': 'c1', 'score': 0.75}]

        class Embedder:
            dimensions = 3

            def embed_query(self, q):
                return [0.0, 0.0, 1.0]

        store = Store()
        out = DenseRetriever(store, Embedder(), [CHUNK]).search('q', filters={'chapter': 2}, top_k=3)
        self.assertEqual(store.args, ('B', 3, {'chapter': 2}))
        self.assertEqual(out[0]['text'], CHUNK['text'])
        self.assertEqual(out[0]['score'], 0.75)


if __name__ == '__main__':
    unittest.main()
