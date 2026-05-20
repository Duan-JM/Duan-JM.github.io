.PHONY: serve
serve:
	cd docs && bundle exec jekyll serve

.PHONY: clean
clean:
	rm -rf ./docs/_site

.PHONY: install
install:
	cd ./docs && bundle config set --local path 'vendor/bundle' && bundle install

