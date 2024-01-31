.PHONY: serve
serve:
	cd docs && bundle exec jekyll serve

.PHONY: clean
clean:
	rm -r ./docs/_site
	rm ./docs/Gemfile.lock

.PHONY: install
install:
	sudo gem install jekyll
	sudo gem install bundler -v 2.4.22
	cd ./docs && sudo bundle install
