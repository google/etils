import logging

from etils.eapp import better_logging


def test_better_logging_idempotent():
  logger = logging.getLogger()

  # Reset possible state from other tests
  if hasattr(logger, "_better_logging_configured"):
    delattr(logger, "_better_logging_configured")

  before = list(logger.handlers)

  better_logging()
  after_first = list(logger.handlers)

  better_logging()
  after_second = list(logger.handlers)

  # First call may add handlers
  assert len(after_first) >= len(before)

  # Second call must not add any more handlers
  assert len(after_second) == len(after_first)
