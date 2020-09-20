.PHONY: serve 
serve:
	cd docs && bundle exec jekyll serve

.PHONY: clean
clean:
	rm -r ./docs/_site
	rm ./docs/Gemfile.lock

.PHONY: install
install:
	gem install jekyll bundler
	cd ./docs && bundle install 

