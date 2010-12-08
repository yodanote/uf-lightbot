LATEX=latex
PDFLATEX=pdflatex
BIBTEX=bibtex

all: report

REPORT_TEX=report.tex
BIB=report

report: report_tex bibliography
	$(LATEX) $(REPORT_TEX)
	$(PDFLATEX) $(REPORT_TEX)

bibliography:report_tex
	$(BIBTEX) $(BIB)
  
report_tex:  
	$(LATEX) $(REPORT_TEX)
