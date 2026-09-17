import unittest
import qwen_phone_gate_probe_20260917 as probe

class PhoneGateTests(unittest.TestCase):
    def test_supervisor_abort_is_not_swallowed_as_http_error(self):
        self.assertFalse(issubclass(probe.ForegroundAbort, Exception))
        with self.assertRaises(probe.ForegroundAbort):
            try:
                raise probe.ForegroundAbort('foreground failure')
            except Exception:
                self.fail('startup swallowed foreground failure')

    def test_exact_label_only(self):
        self.assertEqual(probe.parse_label('LABEL=FILTER_FALSE_POSITIVE'), 'FILTER_FALSE_POSITIVE')
        for raw in ('</think>LABEL=FILTER_FALSE_POSITIVE', 'LABEL=FILTER_FALSE_POSITIVE\nLABEL=FILTER_FALSE_POSITIVE', '', 'FILTER_FALSE_POSITIVE'):
            self.assertEqual(probe.parse_label(raw), 'UNCERTAIN')

    def test_truncation_never_filters(self):
        self.assertEqual(probe.parse_label('LABEL=FILTER_FALSE_POSITIVE', 'length'), 'UNCERTAIN')

    def test_production_rescue_hierarchy(self):
        F, K, U = 'FILTER_FALSE_POSITIVE', 'KEEP_NON_CALL_PHONE_USE', 'UNCERTAIN'
        self.assertEqual(probe.aggregate([F,F,F]), F)
        self.assertEqual(probe.aggregate([F,K]), K)
        self.assertEqual(probe.aggregate([F,F,K]), K)
        self.assertEqual(probe.aggregate([F,U]), U)
        self.assertEqual(probe.aggregate([F,F,U]), U)
        self.assertEqual(probe.aggregate([F]), U)
        self.assertEqual(probe.aggregate([K]), K)

    def test_video_single_request_no_thinking(self):
        r = probe.build_request(b'video', 'prompt')
        self.assertEqual(r['messages'][0]['content'][0]['type'], 'input_video')
        self.assertFalse(r['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(r['max_tokens'], 24)

if __name__ == '__main__':
    unittest.main()
