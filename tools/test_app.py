"""Headless smoke test: runs every dashboard page and fails on any exception.
Run:  python tools/test_app.py"""
import glob, os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
from streamlit.testing.v1 import AppTest
ok = True
for page in ["app.py"] + sorted(glob.glob("pages/*.py")):
    at = AppTest.from_file(os.path.abspath(page), default_timeout=300).run()
    errs = [e.value for e in at.exception]
    print(("FAIL " if errs else "ok   ") + page, errs[:1])
    ok &= not errs
# Also exercise the "store" roll-up path on the explorer page.
at = AppTest.from_file(os.path.abspath(sorted(glob.glob("pages/1_*.py"))[0]), default_timeout=300).run()
at.radio[0].set_value("Total (all stores)").run()
print("total rollup:", "FAIL" if at.exception else "ok", [e.value for e in at.exception][:1])
ok &= not at.exception
sys.exit(0 if ok else 1)
