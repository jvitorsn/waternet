\LoadClass[12pt]{article}
\usepackage[square,numbers]{natbib}
\bibliographystyle{abbrvnat}

\RequirePackage[utf8]{inputenc}
%\RequirePackage[portuguese,brazil]{babel}
\RequirePackage{titlesec}
\RequirePackage{titling}
\RequirePackage{graphicx}
\RequirePackage{indentfirst}
\RequirePackage{float}
\RequirePackage{setspace} 
\RequirePackage{afterpage}
\RequirePackage{tabularx}
\RequirePackage{fancyhdr}
\RequirePackage[labelfont=bf,textfont=bf]{caption}
\setcounter{secnumdepth}{4}
\setcounter{tocdepth}{4}

\usepackage{blindtext}
\usepackage{amsmath}
\usepackage{url}

% Define paginas em branco
\newcommand\blankpage{%
    \null
    \thispagestyle{empty}%
    \addtocounter{page}{-1}%
    \newpage}

% Define as margens
\RequirePackage[a4paper, left = 1.18in, right = 0.79in, top = 1.18in, bottom = 0.79in]{geometry}

\newcommand{\MONTH}{%
  \ifcase\the\month
  \or Janeiro % 1
  \or Fevereiro % 2
  \or Mar\c{c}o % 3
  \or Abril % 4
  \or Maio % 5
  \or Junho % 6
  \or Julho % 7
  \or Agosto % 8
  \or Setembro % 9
  \or Outubro % 10
  \or Novembro % 11
  \or Dezembro % 12
  \fi}

%Formata parágrafo%
\parindent=0.49in
\parskip=6pt
\onehalfspacing

%Referencias %

  
