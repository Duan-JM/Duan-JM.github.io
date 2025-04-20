.PHONY: serve
serve:
	cd docs && bundle exec jekyll serve

.PHONY: clean
clean:
	rm -r ./docs/_site
	rm ./docs/Gemfile.lock

.PHONY: install
install:
	sudo gem install bundler --user -V
	cd ./docs && sudo bundle install

