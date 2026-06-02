MAIN := report
OUTDIR := .build
STRAY := report.log report.synctex.gz report.fdb_latexmk \
	report.fls report.run.xml report.toc report.aux \
	report.bcf report.bbl report.blg

.PHONY: pdf clean

pdf:
	latexmk -pdf -interaction=nonstopmode -file-line-error -outdir=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf $(MAIN).pdf
	rm -f $(STRAY)

clean:
	latexmk -C -outdir=$(OUTDIR) $(MAIN).tex 2>/dev/null || true
	rm -rf $(OUTDIR) $(MAIN).pdf $(STRAY)
