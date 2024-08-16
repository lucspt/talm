# Contributing

We are excited for you to start contributing! Here are some guidelines:

- [Question or Problem](#have-a-question-or-problem)
- [Find A Bug?](#find-a-bug)
- [Coding guidelines](#coding-guidelines)
- [Development Setup](#development-setup)

## Have a Question or Problem?

**TL;DR: Message slack, don't be scared to ask for help**

If you have any questions or problems, you can message the slack
channel directly. Whenever it is appropriate, we like to share
messages with the entire team to ensure everyone is on the same page.

**Do NOT be scared to ask for help!**
If you have already tried to answer the question yourself, but
did not find an answer, don't hesitate to ask anyone else that
you deem fit! We are all here to support each other and **WILL NOT**
look down upon a request for help.

## Find A Bug?

If you find a bug in the code, mention it to the team.
Then, you can submit a pull request with a fix.

## Development Setup

We use [Rye](https://rye.astral.sh/guide/) to manage dependencies, if you do not know it, it's great! [Check it out](https://rye.astral.sh/guide/installation/)!

After you install it, you just have to run:

```bash
rye sync --all-features && source ./scripts/prepare
```

You can then run scripts using rye run python script.py.

### Commonly used rye scripts

```bash
# run ruff linter and formatter
rye run fix

# run mypy type checker and ruff linter
rye run lint

# run test w/ coverage report
rye run test
# OR for html report
rye run test-ui
```

## Coding guidelines

### Commit messages

Your commit message should be as descriptive as possible, as
it leads to more readable histories.

Here is the format we use, which follows the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format:

```text
<type of commit>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

For example:

```text
chore(dependencies): bump random-package from 1.10.0 to 1.11.0
```

The type and subject are **mandatory**, the rest, including the scope, are optional.

### Type

When specifying a type, choose from these options:

- **build**: Changes that affect the build system or external dependencies.
- **chore**: Updating tasks etc; no production code change.
- **ci**: Changes to our CI configuration.
- **docs**: Documentation only changes.
- **feat**: A new feature.
- **fix**: A bug fix.
- **perf**: A code change that regards / improves performance.
- **refactor**: A code change that neither fixes a bug nor adds a feature.
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc).
- **test**: Adding missing tests or correcting existing tests.

### Scope

The optional scope should be the names of the project directories it is affecting.

If adding more than one scope, separate them by commas, e.g.`lib,tests`
