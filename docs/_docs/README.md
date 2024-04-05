# Docs instructions

Note that if you are creating a library or framework, you may want to include documentations. You can use an external framework like mkdocs or readthedocs.io, or include it here. 

Jekyll is designed to host docs very well with a lot of options for customization, but it's a bit difficult to get started. You will need to know about the following files:
- `/docs/_config.yml`: This is the config for the entire webpage. In there, you will find specific settings:
    - `include`: As you can notice, we can find "_docs" there. This means that the relative path `_docs` is included, which resolves to `/docs/_docs`. Note we could've called it `_foobar` if we wanted, it doesn't really matter, but we used `_` as a convention.
    - `defaults`: This variable scopes the path `"_docs"` (again, relative to `_config.yml`) that we defined in `include`. It specifies important default values, such as the left sidebar that include a list of pages, and the right sidebar that includes a table of content, or `toc` as abbreviated. You can override those values by specifying in the front matter of a page, e.g. you can edit `./_docs/api.md` and rename by setting `toc_label: "new name"`. Notice also that `nav: sidebars-docs`, this references the navigation layout explained below.
- `/docs/_data/navigation.yml`: This controls any navigation in your page, including the navigation sidebar on the left side. You can find a variable `sidebar-docs`, which specifies the layout of the 
- `/docs/_docs`: This contains all doc-related pages. Any markdown file placed here will automatically receive the defaults specified in `_config.yml`. The name of the files do not matter, as the title and the relative URL are specified in the front matter. For example, we have `_docs/home.md` which we have specified to redirect to `/docs/`, but we could've chosen` /foo/bar` if we wanted.

Read more about navigation sidebars here: 
- https://mmistakes.github.io/minimal-mistakes/layout-sidebar-nav-list/
- https://mmistakes.github.io/minimal-mistakes/layout-sidebar-custom/

Read more about table of contents here: https://mmistakes.github.io/minimal-mistakes/layout-table-of-contents-post/