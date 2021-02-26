(TeX-add-style-hook
 "classnotes"
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
    "sec:orgf218f8e"
    "sec:org312cd6b"
    "sec:org2c62abf"
    "sec:org9c310d5"
    "sec:orgf0b4437"
    "sec:orge58e342"
    "sec:orgce8814d"
    "sec:orga4360df"
    "sec:orgafb5282"
    "sec:orgd2888ed"
    "sec:org220d3f0"
    "sec:orgc5e99a4"
    "sec:org34ebb15"
    "sec:org3bdbeae"
    "sec:org6fc995e"
    "sec:org80a6377"
    "sec:org131e454"
    "sec:org943f12e"
    "sec:orgb757e87"
    "sec:orgb18a119"
    "sec:orgb649d66"
    "sec:org437fe57"
    "sec:org52b85d9"
    "sec:org9a20759"
    "sec:orgdbaa811"
    "sec:orgfff5446"
    "sec:org3d4a37d"
    "sec:org2840815"
    "sec:org520dd60"
    "sec:orgbe78828"
    "sec:org887f2a1"
    "sec:org15a213b"
    "sec:orgb3a621c"
    "sec:org6332563"
    "sec:org4f628cd"
    "sec:org74114a9"
    "sec:orgd94a9c8"
    "sec:orgc953797"
    "sec:org2f09d0d"
    "sec:org6698360"
    "sec:org6874098"
    "sec:orgb4a235f"
    "sec:org50f10dc"
    "sec:org82d99a2"
    "sec:org129ebc9"
    "sec:org9346ffd"
    "first:main"
    "sec:org063dc82"
    "sec:orgba81177"
    "sec:orga2693a8"
    "sec:org394ef50"
    "fig:org4ba9163"
    "sec:org78a9e99"
    "sec:orgfc90928"
    "sec:org0ddb6b2"
    "sec:org499ccc3"
    "sec:orgd1c9003"
    "sec:org859f16b"
    "sec:orgd4a28aa"
    "sec:org029df39"
    "sec:orge9a2ded"
    "fig:orga6d9573"
    "sec:org5b93fc2"
    "fig:orgf01a383"
    "sec:org583c1b3"
    "sec:org55d48b9"))
 :latex)

