.PHONY: build publish release bump-version bump-patch bump-minor bump-major release-auto

VERSION ?= $(shell python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
PART ?= patch

build:
	uv build

publish:
	uv publish

release: build publish
	gh release create v$(VERSION) \
		dist/gluellm-$(VERSION).tar.gz \
		dist/gluellm-$(VERSION)-py3-none-any.whl \
		--title "v$(VERSION)" \
		--notes "Release v$(VERSION)"

bump-version:
	python3 -c 'import pathlib,re,tomllib; p=pathlib.Path("pyproject.toml"); t=p.read_text(); v=tomllib.loads(t)["project"]["version"]; ma,mi,pa=map(int,v.split(".")); part="$(PART)"; \
if part=="major": ma,mi,pa=ma+1,0,0; \
elif part=="minor": mi,pa=mi+1,0; \
elif part=="patch": pa=pa+1; \
else: raise SystemExit(f"Invalid PART: {part}. Use patch, minor, or major."); \
nv=f"{ma}.{mi}.{pa}"; p.write_text(re.sub(r"(?m)^version = \"\\d+\\.\\d+\\.\\d+\"$$", f"version = \"{nv}\"", t, count=1)); print(f"Bumped: {v} -> {nv}")'

bump-patch: PART=patch
bump-patch: bump-version

bump-minor: PART=minor
bump-minor: bump-version

bump-major: PART=major
bump-major: bump-version

release-auto: bump-version
	$(MAKE) release VERSION=$$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
