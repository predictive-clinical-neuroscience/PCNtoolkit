# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
AUTOAPIDIR	  = autoapi
PORT          ?= 8004

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html show show-docs-remote livehtml livehtml-docs-remote

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)
	rm -rf $(AUTOAPIDIR)

html: clean
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/$(BUILDDIR)/html/index.html')" 

show-docs-remote:
	@cd $(BUILDDIR)/html && python -m http.server $(PORT)

livehtml:
	@sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" --open-browser $(SPHINXOPTS) $(O)

livehtml-docs-remote:
	@sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PORT) $(SPHINXOPTS) $(O)