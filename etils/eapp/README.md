## eapp

Absl app/flag utils.

### `eapp.make_flags_parser`

Dataclass flag parser for absl. This allow to define CLI flags through
dataclasses.

Usage:

```python
from absl import app
from etils import eapp


@dataclasses.dataclass
class Args:  # Define `--user=some_user --verbose` CLI flags
  user: str
  verbose: bool = False


def main(args: Args):
  if args.verbose:
    print(args.user)


if __name__ == '__main__':
  app.run(main, flags_parser=eapp.make_flags_parser(Args))
```

This is a wrapper around
[`simple_parsing`](https://github.com/lebrice/SimpleParsing). See documentation
for details.

This is compatible with `absl.flags`, so you can mix `dataclasses` with `FLAGS`
on the same program.

### `eapp.better_logging`

Improve Python logging when running locally.

*   Display Python logs by default (even when user forgot `--logtostderr`),
    without being polluted by hundreds of C++ logs.
*   Cleaner minimal log format (e.g. `I 15:04:05 [main.py:24]:`)
*   Avoid visual artifacts between TQDM & `logging`
*   Clickable `file.py` hyperlinks redirecting to code search (require
    [terminal support](https://gist.github.com/egmontkob/eb114294efbcd5adb1944c9f3cb5feda))

Usage:

```python
if __name__ == '__main__':
  eapp.better_logging()
  app.run(main)
```

Note this has only effect when user run locally and without `--logtostderr`.
