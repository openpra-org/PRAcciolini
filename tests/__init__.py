from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Set, Optional


def _parallel_test_wrapper(cls, test_fn, files: Set[str], max_workers: Optional[int] = None):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(test_fn, file_path): file_path for file_path in files
        }
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            with cls.subTest(file_path=file_path):
                try:
                    success, error = future.result()
                    if not success:
                        cls.fail(f"Test failed for {file_path}: {error}")
                except Exception as e:
                    cls.fail(f"Test raised exception for {file_path}: {str(e)}")
