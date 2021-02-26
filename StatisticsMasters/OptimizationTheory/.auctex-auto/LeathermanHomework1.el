(TeX-add-style-hook
 "LeathermanHomework1"
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
    "sec:orgb484466"
    "sec:org45e9626"
    "sec:org2383883"
    "sec:org49ea7f6"
    "sec:org6d9aacf"
    "sec:orgabe1625"
    "eq:Dustin"
    "sec:org003354e"
    "sec:org1403c32"
    "sec:orgc952f84"
    "sec:org0a01ff6"
    "sec:orgdff08b7"
    "sec:org92aea71"
    "sec:org8cd83c7"
    "sec:orgbac5438"
    "sec:orgb3ba81b"
    "sec:orgf2f5b3e"
    "sec:orgd835fc9"
    "sec:org94902e4"
    "sec:org017b201"))
 :latex)

