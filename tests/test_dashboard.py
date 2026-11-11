"""Exercise the actual Streamlit entry point without a browser or Ollama."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from flowpilot.storage import list_events, list_requests

APP = Path(__file__).resolve().parents[1] / 'app.py'


class DashboardTests(unittest.TestCase):
    def test_intake_review_and_reanalysis(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / 'ui.db'
            with patch.dict(os.environ, {'FLOWPILOT_DB': str(database), 'OPENAI_API_KEY': ''}):
                app = AppTest.from_file(str(APP)).run()
                self.assertFalse(app.exception)
                app.button(key='analyze_REQ-001').click().run()
                self.assertFalse(app.exception)
                row = list_requests(database)[0]
                self.assertEqual(row['status'], 'Pending review')
                app.text_area(key=f"draft_review_REQ-001_{row['revision']}").set_value('Edited response')
                app.button(key=f"FormSubmitter:review_REQ-001_{row['revision']}-Approve proposal").click().run()
                self.assertFalse(app.exception)
                row = list_requests(database)[0]
                self.assertEqual(row['status'], 'Approved')
                self.assertEqual(row['reviewed_reply'], 'Edited response')
                app.button(key='analyze_REQ-001').click().run()
                row = list_requests(database)[0]
                self.assertEqual(row['status'], 'Pending review')
                app.button(key=f"FormSubmitter:review_REQ-001_{row['revision']}-Reject proposal").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(list_requests(database)[0]['status'], 'Rejected')
                self.assertEqual([event['action'] for event in list_events(database)],
                                 ['Rejected', 'Generated recommendation', 'Approved', 'Generated recommendation'])

    def test_recorded_evaluation_renders(self):
        with TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'FLOWPILOT_DB': str(Path(directory) / 'ui.db')}):
                app = AppTest.from_file(str(APP)).run()
                picker = next(radio for radio in app.radio if radio.label == 'Results')
                picker.set_value('Recorded Mac comparison').run()
                self.assertFalse(app.exception)
                engines = app.dataframe[0].value
                self.assertEqual(engines['Engine'].tolist(), ['Offline rules', 'llama3.2:3b'])
                self.assertEqual(engines['Fallbacks'].tolist(), [0, 0])
