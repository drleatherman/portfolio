(TeX-add-style-hook
 "LeathermanHomework3"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
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
    "sec:org5a031bc"
    "sec:orgb4404c1"
    "sec:org4d8a40a"
    "sec:orgb410f77"
    "sec:org99d1a83"
    "sec:org1b9402a"
    "sec:orgc87dcff"
    "sec:org43fb698"
    "sec:orgfd5ff17"
    "sec:orgd3950f7"
    "sec:org6bae1fd"
    "sec:org27b0f3b"
    "sec:org5fb5e36"
    "sec:org1bee55a"
    "sec:orge8dbf12"
    "sec:org7ddbf46"
    "sec:org0bc19f6"
    "sec:org6a9d4b7"
    "sec:orgc756119"
    "sec:orgdf58f98"
    "sec:orgb8ce061"
    "sec:org6a3e66c"
    "sec:orgf35bdc0"
    "sec:orgda0b6be"))
 :latex)

