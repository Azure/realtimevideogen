"""
Tests for ppt_utils.py using mocks for external dependencies
(python-pptx, LibreOffice via subprocess, and PyMuPDF/fitz).
"""

import logging
import sys
import pytest

from unittest.mock import MagicMock, patch

sys.path.append("wrapper")

from ppt_utils import get_num_slides
from ppt_utils import pptx_to_images


# ---------------------------------------------------------------------------
# get_num_slides
# ---------------------------------------------------------------------------

class TestGetNumSlides:
    """Tests for ppt_utils.get_num_slides()."""

    def _make_slide(self, show: str | None = None) -> MagicMock:
        """Create a mock slide object."""
        slide = MagicMock()
        slide._element.get.return_value = show
        return slide

    def test_counts_visible_slides(self) -> None:
        """Only visible slides are counted when count_hidden=False."""
        slides = [
            self._make_slide(show=None),   # visible
            self._make_slide(show=None),   # visible
            self._make_slide(show="0"),    # hidden
        ]
        presentation = MagicMock()
        presentation.slides = slides

        with patch("ppt_utils.Presentation", return_value=presentation):
            count = get_num_slides("dummy.pptx", count_hidden=False)

        assert count == 2

    def test_counts_all_slides_with_count_hidden_true(self) -> None:
        """All slides (including hidden) are counted when count_hidden=True."""
        slides = [
            self._make_slide(show=None),   # visible
            self._make_slide(show="0"),    # hidden
            self._make_slide(show="0"),    # hidden
        ]
        presentation = MagicMock()
        presentation.slides = slides

        with patch("ppt_utils.Presentation", return_value=presentation):
            count = get_num_slides("dummy.pptx", count_hidden=True)

        assert count == 3

    def test_empty_presentation(self) -> None:
        """Empty presentation returns 0."""
        presentation = MagicMock()
        presentation.slides = []

        with patch("ppt_utils.Presentation", return_value=presentation):
            count = get_num_slides("dummy.pptx")

        assert count == 0

    def test_all_hidden_returns_zero_by_default(self) -> None:
        """All-hidden slides return 0 when count_hidden=False."""
        slides = [
            self._make_slide(show="0"),
            self._make_slide(show="0"),
        ]
        presentation = MagicMock()
        presentation.slides = slides

        with patch("ppt_utils.Presentation", return_value=presentation):
            count = get_num_slides("dummy.pptx")

        assert count == 0

    def test_passes_path_to_presentation(self) -> None:
        """The PPTX path is forwarded to pptx.Presentation."""
        presentation = MagicMock()
        presentation.slides = []

        with patch("ppt_utils.Presentation", return_value=presentation) as mock_pres:
            get_num_slides("/some/path/slides.pptx")

        mock_pres.assert_called_once_with("/some/path/slides.pptx")


# ---------------------------------------------------------------------------
# pptx_to_images
# ---------------------------------------------------------------------------

class TestPptxToImages:
    """Tests for ppt_utils.pptx_to_images()."""

    def _make_subprocess_result(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
    ) -> MagicMock:
        result = MagicMock()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    def _make_fitz_doc(self, num_pages: int = 2) -> MagicMock:
        """Create a minimal mock fitz.Document."""
        doc = MagicMock()
        doc.__len__.return_value = num_pages
        rect = MagicMock()
        rect.width = 1280.0
        rect.height = 800.0
        doc.__getitem__.return_value = MagicMock(rect=rect)

        pages = []
        for _ in range(num_pages):
            page = MagicMock()
            pix = MagicMock()
            page.get_pixmap.return_value = pix
            pages.append(page)

        doc.load_page = MagicMock(side_effect=pages)
        return doc

    def test_raises_on_libreoffice_failure(self, tmp_path) -> None:
        """RuntimeError is raised when LibreOffice returns non-zero exit code."""
        pptx_path = str(tmp_path / "slides.pptx")
        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(returncode=1)):
            with pytest.raises(RuntimeError, match="Failed to generate images from PPTX"):
                pptx_to_images(pptx_path, str(tmp_path))

    def test_raises_when_pdf_not_found(self, tmp_path) -> None:
        """FileNotFoundError is raised when the expected PDF is not produced."""
        pptx_path = str(tmp_path / "slides.pptx")
        # LibreOffice succeeds but the PDF does not exist on disk.
        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(returncode=0)):
            with pytest.raises(FileNotFoundError, match="Expected PDF file not found"):
                pptx_to_images(pptx_path, str(tmp_path))

    def test_returns_image_paths_for_each_page(self, tmp_path) -> None:
        """Returns one image path per PDF page."""
        num_pages = 3
        pptx_path = str(tmp_path / "slides.pptx")

        # Create a dummy PDF file so os.path.exists passes.
        (tmp_path / "slides.pdf").touch()

        fitz_doc = self._make_fitz_doc(num_pages=num_pages)

        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(returncode=0)), \
             patch("ppt_utils.fitz.open", return_value=fitz_doc), \
             patch("ppt_utils.fitz.Matrix", return_value=MagicMock()):
            image_paths = pptx_to_images(pptx_path, str(tmp_path))

        assert len(image_paths) == num_pages
        for i, path in enumerate(image_paths, start=1):
            assert path.endswith(f"slide_{i:03d}.png")

    def test_libreoffice_command_uses_correct_args(self, tmp_path) -> None:
        """The LibreOffice command includes --headless, --convert-to pdf, and the file path."""
        pptx_path = str(tmp_path / "slides.pptx")

        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(returncode=0)) as mock_run:
            # Will raise FileNotFoundError (PDF not created) – that's fine here.
            try:
                pptx_to_images(pptx_path, str(tmp_path))
            except FileNotFoundError:
                pass

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "libreoffice" in cmd
        assert "--headless" in cmd
        assert "--convert-to" in cmd
        assert "pdf" in cmd
        assert pptx_path in cmd

    def test_logger_receives_stdout_and_stderr(self, tmp_path) -> None:
        """When a logger is passed, LibreOffice stdout and stderr are forwarded."""
        pptx_path = str(tmp_path / "slides.pptx")
        mock_logger = MagicMock(spec=logging.Logger)

        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(
                       returncode=1, stdout="out text", stderr="err text")):
            try:
                pptx_to_images(pptx_path, str(tmp_path), logger=mock_logger)
            except RuntimeError:
                pass

        mock_logger.debug.assert_called()
        mock_logger.warning.assert_called()

    def test_custom_width_and_height_applied(self, tmp_path) -> None:
        """Custom width/height are used to build the fitz.Matrix scale factors."""
        num_pages = 1
        pptx_path = str(tmp_path / "deck.pptx")
        (tmp_path / "deck.pdf").touch()

        fitz_doc = self._make_fitz_doc(num_pages=num_pages)

        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(returncode=0)), \
             patch("ppt_utils.fitz.open", return_value=fitz_doc), \
             patch("ppt_utils.fitz.Matrix") as mock_matrix:
            pptx_to_images(pptx_path, str(tmp_path), width=640, height=400)

        # Matrix should have been called with the scale factors derived from
        # the requested dimensions divided by the doc page dimensions.
        mock_matrix.assert_called_once()
        args = mock_matrix.call_args[0]
        assert len(args) == 2   # (matrix_width, matrix_height)
        assert args[0] == pytest.approx(640 / 1280.0)
        assert args[1] == pytest.approx(400 / 800.0)

    def test_no_width_height_uses_dpi(self, tmp_path) -> None:
        """When width=None and height=None, a DPI-based fitz.Matrix is used."""
        num_pages = 1
        pptx_path = str(tmp_path / "deck.pptx")
        (tmp_path / "deck.pdf").touch()

        fitz_doc = self._make_fitz_doc(num_pages=num_pages)

        with patch("ppt_utils.subprocess.run",
                   return_value=self._make_subprocess_result(returncode=0)), \
             patch("ppt_utils.fitz.open", return_value=fitz_doc), \
             patch("ppt_utils.fitz.Matrix") as mock_matrix:
            pptx_to_images(pptx_path, str(tmp_path), width=None, height=None, dpi=144)

        # DPI/72 = 144/72 = 2.0 scale factor
        mock_matrix.assert_called_once_with(2.0, 2.0)
