from scripts.okf_lint import lint_bundle
from scripts.wiki_lint_scan import lint_docs


def test_private_docs_do_not_affect_public_lint(tmp_path):
    (tmp_path / "INDEX.md").write_text("---\ntype: Index\n---\n", encoding="utf-8")
    private = tmp_path / "private"
    private.mkdir()
    (private / "draft.md").write_text("---\ninvalid: [", encoding="utf-8")
    assert lint_docs(tmp_path)["total"] == 1
    report = lint_bundle(tmp_path, profile="minimal")
    assert report.total_md == 1
    assert not report.findings


def test_public_link_to_private_file_is_reported_even_when_present(tmp_path):
    (tmp_path / "INDEX.md").write_text("[local](private/draft.md)", encoding="utf-8")
    (tmp_path / "private").mkdir()
    (tmp_path / "private/draft.md").write_text("Local draft", encoding="utf-8")
    assert lint_docs(tmp_path)["broken_docs"] == [("INDEX.md", "private/draft.md")]
