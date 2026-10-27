project = 'CHART'
author = 'Genentech, Inc.'
copyright = '2025, Genentech, Inc.'

extensions = ['myst_nb']
myst_enable_extensions = ['alert', 'colon_fence', 'dollarmath']
# The tutorial needs its download and minutes of compute, so it is committed
# with its outputs rather than run by the build.
nb_execution_mode = 'off'
exclude_patterns = ['_build']
highlight_language = 'none'

html_theme = 'pydata_sphinx_theme'
html_title = 'CHART'
html_theme_options = {'github_url': 'https://github.com/Genentech/CHART',
                      'header_links_before_dropdown': 10}
html_sidebars = {'**': []}
html_show_sourcelink = False
