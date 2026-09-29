# Legacy code (archived, not run)

The push-to-talk skeleton this repo started with, before any model was
trained, plus three scripts the final system no longer needs. Kept so the
project's history can be read without digging through git.

These files are outside `src/`, so they can't be imported from the `vcm`
package, and `pytest` doesn't collect their tests. They would need their
old imports restored to run. What each file was and what replaced it is in
[../AUDIT.md](../AUDIT.md#8-legacy-code-in-legacy_code).
