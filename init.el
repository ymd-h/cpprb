(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)

(package-refresh-contents)
(package-install 'ox-hugo)

(with-eval-after-load 'ox
  (require 'ox-hugo))
