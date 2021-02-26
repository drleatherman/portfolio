(TeX-add-style-hook
 "classnotes"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
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
    "sec:org0bd9fc0"
    "sec:org7c276ae"
    "sec:orgccb0c78"
    "sec:org999aeae"
    "sec:orgd0544b9"
    "sec:org99e54db"
    "sec:org606260c"
    "sec:org82085b2"
    "sec:org1f998b8"
    "sec:orgd3dd060"
    "sec:org1d4333e"
    "sec:org55d4e05"
    "sec:org272ce08"
    "sec:org3a88861"
    "sec:orgb4a105d"
    "sec:orgb8cdd6d"
    "sec:orgeec5273"
    "sec:org1feeb1c"
    "sec:orgbc49d1c"
    "sec:org3a1fcbf"
    "sec:org4f34143"))
 :latex)

