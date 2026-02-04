import nbformat
from nbclient import NotebookClient
import pytest

@pytest.mark.parametrize(
    "notebook_path", 
    [
        "doc/basics.ipynb",
        "doc/poorman_test.ipynb",
        "doc/preblur_bosonic.ipynb",
        "doc/preblur_test.ipynb",
    ]
)
def test_notebook_execution(notebook_path, tmp_path):
    nb = nbformat.read(notebook_path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python",
        resources={"metadata": {"path": tmp_path}},
    )

    client.execute()
