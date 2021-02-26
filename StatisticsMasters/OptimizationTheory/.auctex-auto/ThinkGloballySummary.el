(TeX-add-style-hook
 "ThinkGloballySummary"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("ulem" "normalem")))
   (add-to-list 'LaTeX-verbatim-environments-local "minted")
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
    "hyperref"
    "minted")
   (LaTeX-add-labels
    "sec:org200fa0b"
    "sec:org2396412"
    "sec:orgb26bf41"
    "sec:org8fafb44"
    "sec:org1587b14"
    "sec:orgc36deeb"
    "sec:org9ba6a15"
    "sec:orgfca3f36"
    "sec:orgba123c7"
    "sec:org6115d8c"
    "eq:1"
    "sec:orgca3d201"
    "sec:org5b82e5a"
    "eq:2"
    "sec:org56b04d0"
    "sec:org3638f82"
    "sec:orgf7e576f"
    "sec:orgfeebf52"
    "sec:org3299395"
    "sec:org60514f1"
    "sec:org6745020"
    "sec:org82ba933"
    "sec:orge59e5fc"
    "sec:orge00e9a0"
    "sec:org2e7decc"
    "sec:org5cbf4cc"
    "eq:3"
    "eq:4"
    "eq:5"
    "sec:org861a824"
    "eq:6"
    "eq:7"
    "eq:8"
    "eq:9"
    "sec:org519ff88"
    "sec:org8fb6264"))
 :latex)

