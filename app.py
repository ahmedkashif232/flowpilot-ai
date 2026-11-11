"""FlowPilot AI — a demo of retrieval-backed IT ticket triage with human approval."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

import pandas as pd
import streamlit as st

from flowpilot.agent import analyze
from flowpilot.ollama import DEFAULT_MODEL, OllamaError, installed_model
from flowpilot.evaluation import run_evaluation
from flowpilot.policies import ROOT, load_policies
from flowpilot.storage import add_request, import_csv, init_db, list_events, list_requests, record_decision, save_analysis

st.set_page_config(page_title='FlowPilot AI | Workflow Agent', page_icon='🧭', layout='wide', initial_sidebar_state='auto')
init_db()

st.markdown('''
<style>
.block-container {padding-top: 1.8rem; max-width: 1360px;}
[data-testid="stMetric"] {border: 1px solid #28394d; border-radius: 14px; background: #162237; padding: 15px;}
[data-testid="stMetricLabel"] {color:#abc1d6;}
section[data-testid="stSidebar"] {border-right: 1px solid #26384e;}
.hero {padding:24px 28px;border-radius:20px;background:linear-gradient(110deg,#142d42 0%,#152238 55%,#1c2540 100%);border:1px solid #2b5366;margin-bottom:20px;}
.hero h1{color:#f2f8ff;font-size:2.4rem;margin-bottom:4px;letter-spacing:-0.035em}
[data-testid="stText"] pre {white-space:pre-wrap;overflow-wrap:anywhere;}
.hero p{color:#bbcfdf;margin:0;font-size:1rem;}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 🧭 FlowPilot AI')
    st.caption('Service desk triage')
    st.divider()
    st.markdown('### Analysis engine')
    engine = st.radio('Choose a mode', ['Offline rules', 'Local Ollama', 'OpenAI API'],
                      help='Offline is deterministic. Ollama runs an LLM on your own Mac. OpenAI uses a paid external API.')
    llm_requested = engine != 'Offline rules'
    key = os.environ.get('OPENAI_API_KEY', '')
    try:
        key = key or st.secrets.get('OPENAI_API_KEY', '')
    except (FileNotFoundError, KeyError):
        pass
    provider = 'ollama' if engine == 'Local Ollama' else 'openai'
    model = (os.getenv('OLLAMA_MODEL', DEFAULT_MODEL) if engine == 'Local Ollama'
             else os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))
    if engine == 'OpenAI API' and not key:
        st.warning('No API key set. Offline baseline will be used.')
    elif engine == 'Local Ollama':
        try:
            installed_model(model)
            st.success(f'Local model ready · {model}')
        except (OllamaError, ValueError) as exc:
            st.warning(f'{exc} Analysis will fall back to offline rules.')
        st.button('Refresh Ollama status')
    elif engine == 'OpenAI API':
        st.success(f'OpenAI: {model}')
    else:
        st.info('Offline rules + local retrieval')
    st.divider()
    st.caption('Approval records a simulated decision in SQLite.')
    st.caption('Only use synthetic or non-sensitive requests in this portfolio demo. API mode sends request text and retrieved policies to OpenAI.')

requests = list_requests()
statuses = Counter(r['status'] for r in requests)
analyses = [r for r in requests if r['analysis']]

st.markdown('''<div class="hero"><h1>FlowPilot AI</h1>
<p>Review incoming service requests, policy evidence and proposed responses.</p></div>''', unsafe_allow_html=True)
a, b, c, d = st.columns(4)
a.metric('Total requests', len(requests))
b.metric('Awaiting review', statuses['Pending review'])
c.metric('Approved proposals', statuses['Approved'])
d.metric('Analyzed', len(analyses))
st.write('')

t_inbox, t_metrics, t_eval, t_policy, t_audit = st.tabs(['📥 Request inbox', '📊 Workflow analytics', '🧪 Evaluation', '📚 Policy knowledge', '🗂️ Audit trail'])

with t_inbox:
    left, right = st.columns([1.05, 1.95], gap='large')
    with left:
        st.subheader('Incoming requests')
        queue = st.selectbox('Filter by status', ['All', 'New', 'Pending review', 'Approved', 'Rejected'])
        visible = [r for r in requests if queue == 'All' or r['status'] == queue]
        selected = None
        if visible:
            labels = {r['id']: f"{r['id']} · {r['subject'][:34]}" for r in visible}
            selected_id = st.selectbox('Choose a request', [r['id'] for r in visible], format_func=lambda x: labels[x], label_visibility='collapsed')
            selected = next(r for r in requests if r['id'] == selected_id)
            st.text(f"{selected['requester']} · {selected['status']}")
            st.text(selected['subject'])
            st.text(selected['description'])
            label = 'Analyze this request' if not selected['analysis'] else 'Re-analyze and reopen review'
            if st.button(label, key=f'analyze_{selected_id}', type='primary', width='stretch'):
                with st.spinner('Reviewing request and retrieving policies...'):
                    result, _ = analyze(selected['subject'], selected['description'], use_llm=llm_requested, api_key=key, model=model, provider=provider)
                    save_analysis(selected_id, result.to_dict())
                st.rerun()
        else:
            st.info('No requests match this filter.')
        if 'intake_message' in st.session_state:
            st.success(st.session_state.pop('intake_message'))
        st.divider()
        with st.expander('➕ Add a new request'):
            with st.form('new_request', clear_on_submit=True):
                requester = st.text_input('Requester', placeholder='Demo User')
                subject = st.text_input('Subject', placeholder='Cannot access account')
                description = st.text_area('Description', placeholder='Describe the issue...', height=115)
                submitted = st.form_submit_button('Create request', width='stretch')
            if submitted:
                try:
                    request_id = add_request(requester, subject, description)
                    st.session_state.intake_message = f'Created {request_id}.'
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        with st.expander('📥 Import CSV requests'):
            st.caption('Columns: requester, subject, description. Max 100 rows per upload, 12,000 characters per description.')
            upload = st.file_uploader('CSV file', type=['csv'], key='csvupload')
            if upload is not None and st.button('Import requests', width='stretch'):
                try:
                    imported = import_csv(upload.getvalue())
                    st.session_state.intake_message = f'Imported {imported} requests.'
                    st.rerun()
                except (csv.Error, ValueError) as exc:
                    st.error(f'Could not import file: {exc}')
    with right:
        st.subheader('Recommendation workspace')
        if selected and selected['analysis']:
            result = selected['analysis']
            if result.get('warning'):
                st.warning(result['warning'])
            m1, m2, m3 = st.columns(3)
            m1.metric('Category', result['category'])
            m2.metric('Priority', result['priority'])
            m3.metric('Review state', selected['status'])
            st.caption(f"Engine: {result['mode']} · {result['confidence']}")
            st.text(result['summary'])
            st.markdown('#### Recommended action')
            st.markdown(f"**Route to:** {result['owner']}")
            st.text(result['proposed_action'])
            st.markdown('#### Reasoning')
            st.text(result['rationale'])
            st.markdown('#### Retrieved policy evidence')
            ids = result.get('citation_ids', [])
            policies = load_policies()
            policy_by_id = {p['id']: p for p in policies}
            retrieved = result.get('retrieved_ids') or ids
            selected_policies = [policy_by_id[pid] for pid in retrieved if pid in policy_by_id]
            if selected_policies:
                for p in selected_policies:
                    label = 'Cited' if p['id'] in ids else 'Retrieved'
                    with st.expander(f"{p['id']} · {p['title']} · {label}"):
                        st.write(p['text'])
            else:
                st.warning('No supporting policy citations. Check the knowledge base manually.')
            st.markdown('#### Editable response draft')
            draft = selected['reviewed_reply'] if selected['reviewed_reply'] is not None and selected['status'] == 'Approved' else result['draft_reply']
            if selected['status'] == 'Pending review':
                review_key = f"review_{selected_id}_{selected['revision']}"
                with st.form(review_key, enter_to_submit=False):
                    reviewed_text = st.text_area('Reply draft', value=draft, key=f"draft_{review_key}",
                                                 height=185, max_chars=5000)
                    st.caption('Approval saves the edited draft and logs the decision.')
                    yes, no = st.columns(2)
                    approved = yes.form_submit_button('Approve proposal', type='primary', width='stretch')
                    rejected = no.form_submit_button('Reject proposal', width='stretch')
                if approved or rejected:
                    try:
                        record_decision(selected_id, 'Approved' if approved else 'Rejected',
                                        reviewed_text, expected_revision=selected['revision'])
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            else:
                st.text_area('Saved reply' if selected['status'] == 'Approved' else 'Rejected draft',
                             value=draft, key=f"draft_{selected_id}_{selected['revision']}",
                             height=185, disabled=True)
                st.info(f"Decision recorded: {selected['status']}. Re-analyze to generate a new pending proposal.")
            st.download_button('⬇ Export recommendation (JSON)',
                               data=json.dumps({'request_id': selected_id, 'status': selected['status'], 'analysis': result, 'reviewed_reply': selected['reviewed_reply']}, indent=2),
                               file_name=f'{selected_id}_analysis.json', mime='application/json')
        else:
            st.info('Select a request and click **Analyze this request** to generate a policy-supported recommendation.')

with t_metrics:
    st.subheader('Workflow activity')
    st.caption('Counts reflect the latest proposal on each request.')
    if analyses:
        total_reviewed = statuses['Approved'] + statuses['Rejected']
        cx, cy, cz = st.columns(3)
        cx.metric('Approval rate', f'{statuses["Approved"] / total_reviewed:.0%}' if total_reviewed else 'N/A')
        cy.metric('Review decisions', total_reviewed)
        cz.metric('Open recommendations', statuses['Pending review'])
        chart_rows = Counter(r['analysis']['category'] for r in analyses)
        st.markdown('#### Requests analyzed by category')
        st.bar_chart(pd.DataFrame({'Requests': dict(chart_rows)}), color='#70E1B5')
        st.markdown('#### Review queue')
        st.dataframe(pd.DataFrame([{'ID': r['id'], 'Subject': r['subject'],
                                   'Category': r['analysis']['category'], 'Priority': r['analysis']['priority'],
                                   'Status': r['status']} for r in analyses]), hide_index=True, width='stretch')
    else:
        st.info('Run a few analyses to populate operational metrics.')

with t_eval:
    st.subheader('Triage evaluation')
    st.caption('24 synthetic cases with category, priority and one target policy per ticket.')
    options = ['Offline rules', 'Recorded Mac comparison', 'Run on this Mac']
    report_mode = st.radio('Results', options, horizontal=True)
    report_path = ROOT / 'docs' / 'benchmark.json'
    reports = {}
    if report_mode == options[0]:
        reports = {'offline': run_evaluation()}
    elif report_mode == options[1]:
        if report_path.exists():
            reports = json.loads(report_path.read_text(encoding='utf-8'))
        else:
            st.info('No benchmark has been recorded yet.')
    else:
        local_model = os.getenv('OLLAMA_MODEL', DEFAULT_MODEL)
        if st.button(f'Compare with {local_model}'):
            try:
                with st.spinner('Running the local model on 24 cases. This can take several minutes.'):
                    st.session_state.live_report = {
                        'offline': run_evaluation(),
                        'ollama': run_evaluation(engine='ollama', model=local_model),
                    }
            except OllamaError as exc:
                st.error(str(exc))
        reports = st.session_state.get('live_report', {})
    if reports:
        comparison = []
        for report in reports.values():
            comparison.append({
                'Engine': report['model'] or 'Offline rules', 'Cases': report['cases'],
                'Fallbacks': report['fallback_count'],
                'Category accuracy': report['category_accuracy'],
                'Priority accuracy': report['priority_accuracy'],
                'Category macro-F1': report['macro_f1'],
                'Target policy in top 3': report['policy_hit_at_3'],
                'Target policy cited': report['target_citation_rate'],
                'Median latency (s)': report['median_seconds'],
            })
        st.dataframe(pd.DataFrame(comparison), hide_index=True, column_config={
            col: st.column_config.NumberColumn(format='percent')
            for col in ('Category accuracy', 'Priority accuracy', 'Category macro-F1',
                        'Target policy in top 3', 'Target policy cited')
        })
        key_report = st.selectbox('Inspect engine', list(reports))
        metrics = reports[key_report]
        model_only = metrics['engine_only']
        if metrics['fallback_count']:
            count = model_only['cases']
            st.warning(f"{metrics['fallback_count']} fallback cases. Model-only metrics use {count} completed cases.")
            if count:
                st.caption(f"Model-only category {model_only['category_accuracy']:.1%} · "
                           f"priority {model_only['priority_accuracy']:.1%} · macro-F1 {model_only['macro_f1']:.3f}")
        mistakes = st.checkbox('Show category or priority mismatches only')
        rows = metrics['rows']
        if mistakes:
            rows = [r for r in rows if not (r['Category correct'] and r['Priority correct'])]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        with st.expander('Per-category metrics'):
            st.dataframe(pd.DataFrame(metrics['by_category']).T)
        st.download_button('Download complete results (JSON)', json.dumps(reports, indent=2),
                           'flowpilot_evaluation.json', 'application/json')
        st.caption('Policy hits measure overlap with the labeled policy ID. Draft quality and real-world accuracy require separate evaluation.')

with t_policy:
    st.subheader('Local policy knowledge base')
    st.write('FlowPilot ranks policy snippets by keyword overlap and gives the top three matches to the selected analysis engine.')
    for policy in load_policies():
        with st.expander(f"{policy['id']} · {policy['title']} ({policy['category']})"):
            st.write(policy['text'])
    st.caption('All policies and example requests describe a fictional demonstration company and must be reviewed before any real-world use.')

with t_audit:
    st.subheader('Decision audit trail')
    st.caption('Append-only action events for this demo. Approvals are logged; no external system is changed.')
    events = list_events()
    if events:
        st.dataframe(pd.DataFrame(events)[['at', 'request_id', 'actor', 'action', 'details']], hide_index=True, width='stretch')
    else:
        st.info('No events recorded yet.')
