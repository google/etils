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
