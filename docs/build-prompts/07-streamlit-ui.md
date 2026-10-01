# 07: Streamlit UI

**Effort:** medium · **Prereqs:** 06 · **Design refs:** §5 (F1–F7), §8 (status behaviour), §11 (behaviour table), §18 (demo)

````text
Read docs/design.md (§5, §8, §11, §18) and docs/addendum.md, plus docs/progress.md.

Task: a clean Streamlit UI on top of the API. The demo runs on this, so it should look tidy and never crash.

Build:
- Sidebar or page: PDF upload -> POST /documents; a documents list that polls GET /documents every ~2 s
  while anything is in progress, showing status badges (QUEUED / PROCESSING n/N / PARTIAL n/N / READY / FAILED + reason)
  and a progress bar.
- Ask page: chat-style input; render the /query response:
  - DOCUMENT: the answer, citation chips like "Report.pdf · p.47 (PDF p.53)" that expand to show the snippet,
    and a "⚠ number not found verbatim in source" badge when number_check failed.
  - GENERAL: the answer with the general-knowledge label clearly visible.
  - MIXED: two sections, "📄 From your documents" and "🌐 General knowledge".
  - Abstentions: the reason, the closest pages, and the partial-coverage note when present.
  - The API being down or slow -> a friendly message, not a stack trace.
- 👍/👎 buttons per answer -> a new POST /feedback {trace_id, value} endpoint that updates the requests row
  (add the endpoint and a test for it).
- Keep it single-turn (history display is fine, but no conversation memory is sent to the backend).
- Leave a placeholder "Metrics" page; step 08 fills it.

Keep UI code simple: a small API-client module plus page files. Unit-test the API-client/formatting helpers;
the Streamlit layout itself doesn't need tests.

When done: run tests, describe how to click through the main flows, append a "Step 07" entry to
docs/progress.md, and stop.
````
