# How I plan the next work (Scrum-style)

This is the plan for the next sprints of **TrustRUL**. It is a planned way of working for a one-person project, not a record of past team work.

## Working agreement
- Sprint: one week. Sprint goal: one backlog item finished, with its numbers in `reports/` and the README.
- Planning on Monday (pick items), review on Friday (open the project page, check the numbers), a short retrospective note in the commit log.
- Board: one GitHub issue per backlog item, columns To do, In progress, Done (GitHub Projects).
- Branches: `main` stays working; each item gets its own branch and a pull request.

## Definition of done
1. The script runs in a fresh virtual environment from `requirements.txt`.
2. Every number in the README and on the page matches `reports/`.
3. The project page opens without console errors and works on a phone.
4. No secrets or personal data in the files or history.
5. Limits are stated next to the result.

## Product backlog
| # | User story | Acceptance |
|---|---|---|
| 1 | As a reviewer, I want the GRU's interval calibrated with conformal prediction, so that its stated 90 percent range is honest. | Retrain with src/export_artifacts.py, save validation predictions, apply the split-conformal step; coverage within 3 points of 90 on the test engines. |
| 2 | As a planner, I want the results on FD002 to FD004, so that I know how the model copes with several operating conditions. | Same script runs on each subset; a table in the README. |
| 3 | As a reader, I want an expert to check the EU AI Act note in the model card, so that the legal statement is reliable. | Note reviewed or removed. |
| 4 | As a planner, I want the review flag compared with a simple rule, so that I know whether it helps. | Flagged and unflagged error compared with a bootstrap interval. |
