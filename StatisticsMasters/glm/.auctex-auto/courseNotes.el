(TeX-add-style-hook
 "courseNotes"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "inputenc"
    "fontenc"
    "graphicx"
    "grffile"
    "longtable"
    "wrapfig"
    "rotating"
    "ulem"
    "amsmath"
    "textcomp"
    "amssymb"
    "capt-of"
    "hyperref")
   (LaTeX-add-labels
    "sec:orga262ebd"
    "sec:org82cf2e7"
    "sec:org2cd6c90"
    "sec:org0733c8d"
    "sec:org1accf42"
    "sec:orge281343"
    "sec:orgf452836"
    "sec:org7a4fd6a"
    "sec:orga9ad612"
    "sec:orgba142e2"
    "sec:orga316ea4"
    "sec:orgd8243fc"
    "sec:org607952a"
    "sec:org9e30212"
    "sec:org32cec25"
    "sec:org9a3ed04"
    "sec:org7d17b4b"
    "sec:org79855b5"
    "sec:org2b73379"
    "sec:orgf34979b"
    "sec:orga28ea07"
    "sec:org1082f63"
    "sec:org703ef13"
    "sec:orgd08bfa8"
    "sec:orgd679246"
    "sec:orgab713aa"
    "sec:orgf7377ea"
    "sec:orgcdd71af"
    "sec:orgb3de8a0"
    "sec:org0658e7f"
    "sec:org3c9652d"
    "sec:orgf0c5338"
    "sec:org1c039cd"
    "sec:orgbc167d7"
    "sec:orgcf6bfb0"))
 :latex)

