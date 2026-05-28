.PHONY: serve
serve:
	cd docs && PATH="/opt/homebrew/opt/ruby@3.2/bin:$$PATH" bundle exec jekyll serve

.PHONY: clean
clean:
	rm -rf ./docs/_site

.PHONY: install
install:
	cd ./docs && PATH="/opt/homebrew/opt/ruby@3.2/bin:$$PATH" bundle config set --local path 'vendor/bundle' && PATH="/opt/homebrew/opt/ruby@3.2/bin:$$PATH" bundle install

