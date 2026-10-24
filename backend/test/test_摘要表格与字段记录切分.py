import json
import unittest
from unittest.mock import AsyncMock, patch

from service.context_policy import NLISettings
from service.summary_evidence_units import split_evidence_units, structured_windows, render_spans, validate_spans
from service.summary_sentence_evidence import sentence_catalog, selected_context, verify_relevant_claims
from service.summary_message_verification import verify_message_claims_sync
from test_摘要相关证据句选择 import context, claim, Counter, answer
from test_摘要消息编号直接核验 import Model


HEADER = '| 编号 | 数值 |\n| --- | --- |'
TABLE = HEADER + '\n' + '\n'.join(f'| A{i} | {i}% |' for i in range(12))


def units(text, limit=10000):
    return split_evidence_units(text, 0, len(text), lambda s: len(s) <= limit)


class StructuredUnitTests(unittest.IsolatedAsyncioTestCase):
    def test_short_table_is_whole_and_sentences_inside_cells_are_not_split(self):
        text = '| 模型 | 说明 |\n| --- | --- |\n| A | 结果准确。条件明确。 |'
        got = units(text)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['unit_type'], 'table')
        self.assertEqual(got[0]['text'], text)

    def test_long_table_repeats_header_without_splitting_or_losing_rows(self):
        got = units(TABLE, 62)
        self.assertGreater(len(got), 1)
        seen = []
        for u in got:
            self.assertTrue(u['text'].startswith(HEADER + '\n'))
            self.assertLessEqual(len(u['text']), 62)
            validate_spans(TABLE, u)
            seen.extend(TABLE[s['start']:s['end']] for s in u['body_rows'])
        self.assertEqual(seen, TABLE.splitlines()[2:])

    def test_caption_units_and_notes_are_kept_in_every_subtable(self):
        text = '表1：准确率（%）\n' + TABLE + '\n注：数据来自测试集D。'
        got = units(text, 80)
        self.assertGreater(len(got), 1)
        self.assertTrue(all(u['text'].startswith('表1：准确率（%）\n') and u['text'].endswith('注：数据来自测试集D。') for u in got))

    def test_markdown_optional_outer_pipes(self):
        text = '编号 | 数值\n--- | ---\nA | 90%\nB | 80%'
        self.assertEqual(units(text)[0]['text'], text)
        self.assertEqual(units(text)[0]['unit_type'], 'table')

    def test_escaped_and_inline_code_pipes_do_not_change_column_mapping(self):
        text = '| 模型 | 结果 |\n| --- | --- |\n| `A|B` | 90% |\n| A\\|B | 80% |'
        got = units(text)
        self.assertEqual(len(got[0]['body_rows']), 2)

    def test_repeated_header_tables_are_separate(self):
        short = HEADER + '\n| A | 90% |'
        got = units(short + '\n\n' + short)
        self.assertEqual([u['unit_type'] for u in got], ['table', 'table'])

    def test_header_only_and_column_mismatch_fail_without_guessing(self):
        self.assertEqual(units(HEADER)[0]['body_rows'], [])
        with self.assertRaises(ValueError): units(HEADER + '\n| A | 90% | 多余列 |')

    def test_escaped_newlines_in_tool_json_keep_raw_positions(self):
        text = HEADER.replace('\n', r'\\n') + r'\\n\\t' + '| A | 90% |' + r'\\n\\t' + '| B | 80% |'
        got = units(text)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]['text'], HEADER + '\n| A | 90% |\n| B | 80% |')
        for s in got[0]['source_spans']:
            self.assertIn(text[s['start']:s['end']], got[0]['text'])

    def test_repeated_key_value_records_keep_empty_fields_and_dates_separate(self):
        text = (r'| Status: | Approved | | Approval date: | 30/03/2026 | | Contact name: | | |'
                + r'\\n\\t\\t' + r'| Status: | Approved | | Approval date: | 02/04/2026 | | Contact name: | | |')
        got = units(text)
        self.assertEqual([u['unit_type'] for u in got], ['record', 'record'])
        self.assertIn('30/03/2026', got[0]['text']); self.assertNotIn('02/04/2026', got[0]['text'])
        self.assertIn('02/04/2026', got[1]['text'])
        self.assertTrue(all('Contact name:' in u['text'] for u in got))

    def test_explicit_records_are_not_mixed_and_id_repeats_in_long_record(self):
        text = ('记录1：\n试验编号：A001\n状态：Approved\n日期：2026-03-30\n联系人：李某\n'
                '记录2：\n试验编号：A002\n状态：Approved\n日期：2026-04-02')
        got = units(text, 38)
        self.assertGreater(len(got), 2)
        for u in got:
            self.assertLessEqual(len(u['text']), 38)
            self.assertFalse('A001' in u['text'] and 'A002' in u['text'])
            self.assertTrue(u['text'].startswith('记录'))
            self.assertTrue('试验编号' in u['text'])

    def test_repeated_identity_starts_new_record_without_inheriting_previous_id(self):
        text = 'Main ID: A001\nStatus: Approved\nMain ID: A002\nStatus: Rejected'
        got = units(text)
        self.assertEqual(len(got), 2)
        self.assertNotIn('A001', got[1]['text'])

    def test_registry_identity_is_not_assigned_to_preceding_status_history(self):
        text = '| Status: | Approved | Approval date: | 30/03/2026 |\n| Register: | ClinicalTrials.gov | Main ID: | NCT001 |'
        got = units(text)
        self.assertEqual(len(got), 2)
        self.assertNotIn('NCT001', got[0]['text'])
        self.assertNotIn('30/03/2026', got[1]['text'])

    def test_webpage_description_is_not_a_record_field(self):
        text = 'Visit the registry. | Main ID: | A001 | Status: | Approved |'
        got = units(text)
        record = next(u for u in got if u['unit_type'] == 'record')
        self.assertNotIn('Visit the registry', record['text'])
        self.assertIn('A001', record['text'])

    def test_inline_field_values_and_section_heading_are_not_previous_field_values(self):
        text = '| Main ID: | A001 | Affiliation: | | Key inclusion & exclusion criteria | Inclusion Criteria: Age >= 18. | Exclusion Criteria: Prior stroke. |'
        got = units(text)
        fields = [f for u in got for f in u['body_rows']]
        affiliation = next(f for f in fields if f['key'] == 'affiliation')
        self.assertNotIn('Inclusion', text[affiliation['start']:affiliation['end']])
        self.assertTrue(any(f['key'] == 'inclusion criteria' for f in fields))
        self.assertTrue(any(f['key'] == 'exclusion criteria' for f in fields))

    def test_ordinary_prose_and_windows_path_remain_exact(self):
        text = r'模型准确率90%。路径为C:\new\table。'
        got = units(text)
        self.assertEqual([u['unit_type'] for u in got], ['sentence', 'sentence'])
        self.assertEqual(''.join(u['text'] for u in got), text)

    def test_single_oversize_table_row_is_explicit_error(self):
        with self.assertRaises(ValueError): units(HEADER + '\n| A | ' + '长' * 200 + ' |', 62)

    def test_long_field_repeats_label_and_id_and_preserves_every_value_character(self):
        text = 'Main ID: A001\nDescription: ' + '长' * 200
        got = units(text, 62)
        self.assertGreater(len(got), 1)
        values = []
        for u in got:
            self.assertLessEqual(len(u['text']), 62)
            self.assertIn('Main ID: A001', u['text']); self.assertIn('Description:', u['text'])
            self.assertTrue(u['field_fragmented'])
            validate_spans(text, u)
            values.extend(text[s['start']:s['end']] for s in u['body_rows'])
        self.assertEqual(''.join(values), text[text.index('Description:') + len('Description:'):])

    def test_long_field_repacking_does_not_lose_its_label(self):
        text = 'Main ID: A001\nDescription: ' + '词 ' * 100
        initial = units(text, 120)
        second = [v for u in initial for v in structured_windows(u, text, lambda s: len(s) <= 62)]
        self.assertTrue(all('Description:' in u['text'] and 'Main ID: A001' in u['text'] for u in second))
        self.assertTrue(all(len(u['text']) <= 62 for u in second))

    def test_second_nli_pack_keeps_header_and_all_rows(self):
        original = units(TABLE)[0]
        got = structured_windows(original, TABLE, lambda s: len(s) + len(claim()['claim']) + 2 <= 80)
        self.assertGreater(len(got), 1)
        seen = []
        for u in got:
            self.assertTrue(u['text'].startswith(HEADER))
            self.assertLessEqual(len(u['text']) + len(claim()['claim']) + 2, 80)
            seen.extend(u['body_rows'])
        self.assertEqual(seen, original['body_rows'])

    def test_subtable_provenance_is_multiple_spans_not_false_contiguous_quote(self):
        got = units(TABLE, 62)
        last = got[-1]
        self.assertEqual(last['quote_kind'], 'assembled_source_spans')
        self.assertNotEqual(TABLE[last['start']:last['end']], last['text'])
        self.assertEqual(render_spans(TABLE, last['source_spans']), last['text'])

    def test_missing_header_is_not_fetched_from_unprovided_context(self):
        ctx = context(TABLE, [{'start': len(HEADER) + 1, 'text': TABLE[len(HEADER) + 1:]}])
        catalog = sentence_catalog(claim(), ctx, fits=lambda s: len(s) < 62)
        self.assertTrue(all(u['unit_type'] == 'table' and u['header_status'] == 'absent' for u in catalog))
        self.assertTrue(all(not u['fixed_prefix'] for u in catalog))
        self.assertTrue(all(u['start'] >= len(HEADER) + 1 for u in catalog))

    def test_selected_subtable_does_not_add_unselected_rows(self):
        ctx = context(TABLE); catalog = sentence_catalog(claim(), ctx, fits=lambda s: len(s) <= 62)
        last = catalog[-1]; narrowed = selected_context(ctx, [last])
        self.assertEqual(narrowed['sources']['m'][0]['text'], last['text'])
        self.assertNotIn('| A0 |', narrowed['sources']['m'][0]['text'])

    def test_tampered_assembled_text_and_out_of_scope_header_are_rejected(self):
        ctx = context(TABLE); catalog = sentence_catalog(claim(), ctx, fits=lambda s: len(s) <= 62)
        bad = {**catalog[0], 'text': '改写后的表格'}
        with self.assertRaises(ValueError): selected_context(ctx, [bad])
        restricted = context(TABLE, [{'start': len(HEADER) + 1, 'text': TABLE[len(HEADER) + 1:]}])
        with self.assertRaises(ValueError): selected_context(restricted, [catalog[0]])

    def test_nli_receives_header_on_every_window_and_reports_composite_provenance(self):
        ctx = context(TABLE); catalog = sentence_catalog(claim(), ctx, fits=lambda s: len(s) < 10000)
        narrowed = selected_context(ctx, catalog)
        with patch('service.nli_verifier.get_nli', return_value=Model()):
            report = verify_message_claims_sync([claim()], narrowed, NLISettings(max_length=80))
        self.assertGreater(report['window_count'], 1)
        for w in report['all_windows']:
            self.assertTrue(w['quote'].startswith(HEADER))
            self.assertEqual(render_spans(TABLE, w['source_spans']), w['quote'])
            self.assertLessEqual(w['input_tokens'], 80)
        self.assertTrue(report['claims_with_program_quotes'][0]['references'][0]['source_spans'])

    async def test_structured_units_follow_selector_ids_to_nli(self):
        ctx = context(TABLE)
        call = AsyncMock(side_effect=lambda messages, stage: answer(messages))
        with patch('service.nli_verifier.get_nli', return_value=Model()):
            report = await verify_relevant_claims([claim()], ctx, NLISettings(max_length=80), call, counter=Counter())
        self.assertEqual(report['status'], 'passed')
        self.assertTrue(all(u['unit_type'] == 'table' for u in report['evidence_selection']['sentence_catalog']))
        self.assertTrue(all(w['quote'].startswith(HEADER) for w in report['all_windows']))


if __name__ == '__main__': unittest.main()
