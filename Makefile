.PHONY: serve
serve:
	cd docs && bundle exec jekyll serve

.PHONY: clean
clean:
	rm -rf ./docs/_site

.PHONY: install
install:
	sudo gem install bundler --user -V
	cd ./docs && sudo bundle install

