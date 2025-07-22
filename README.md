# Qwen 博客

博客基于[Hugo](https://gohugo.io)编译框架，Hugo可在各平台上方便的安装。

博客主题基于[PaperMod](https://github.com/adityatelange/hugo-PaperMod)，同时借鉴了[VibrantShadows](https://github.com/adityatelange/hugo-PaperMod)，以及[OpenAI](https://openai.com)的设计。

## Hugo安装

推荐使用mac
```bash
brew install hugo #会直接安装最新版本的hugo，安装完编译blog会出错，版本不对齐，执行这个命令会安装go，然后再使用下面命令安装旧版本hugo：
GO_ENABLED=1 go install -tags extended github.com/gohugoio/hugo@release-0.124.0
brew uninstall hugo #删掉高版本hugo,拷贝低版本的hugo到bin
cp /Users/$currentusr/go/bin/hugo /usr/local/bin/
```

Windows直接下载v0.124.1，复制`hugo.exe`到本项目根目录即可，不需要安装go
- https://github.com/gohugoio/hugo/releases/download/v0.124.1/hugo_extended_0.124.1_windows-amd64.zip

## 编译

编译网站，默认输出到`public`文件夹中。
```bash
hugo 
```

Hugo也内置了本地服务器，用于调试功能。
```bash
hugo server -D
```
其中`-D`声明也需要编译标记为draft的页面。


## 新建博客

```bash
hugo new blog/<path>
```
将在`content/blog/<path>`根据archetypes中的骨架新建文档，注意需要路径需要以`.md`结尾。

可以按照文件夹管理，即`path`为`<name>/index.md`，也可以直接在目录下建立文档`<name>.md`。
对于中文页面，只需要添加额外的后缀`.zh`即可，即文档后缀为`.zh.md`。

本博客中普通页面的frontmatter包含大量自定义配置，其中`header`组为页面头部配置，`cover`组为页面封面配置。配置名是自解释的。

注意：
- 如果内容开头为button或分级标题，请手动设置`summary`
- 如需支持latex math，请设置`math: true`，文中`$`、`$$`、`\(`、`\[`标志的内容将使用katex自动解析
- 如需支持样例库，请设置`gallery: true`，并配合example shortcode实现

### shortcode

hugo提供了shortcode功能，shortcode是代码片段的模板，可以快速完成内容排版。

内置的figure shortcode用于图片排版
```
{{< figure src="" title="" class="" width="" height="" >}}
```
本模板的CSS中对于以`#center`结尾的`img`，自动居中。因而以下代码可以快速居中图片
```
{{< figure src="test.jpg#center" >}}
```
搭配本模板中的`wide` class可实现宽版图片，搭配本模板中的`gallery` class可实现阴影效果。

内置的highlight shortcode用于代码高亮（注意fenced code blocks，即`\`\`\``围绕的代码段也是可用的）
```
{{< highlight python3 >}}
print("hello world!")
{{< /highlight >}}
```

更多内置shortcode，如youtube, tweet, vimeo, instagram等，请见[官方说明](https://gohugo.io/content-management/shortcodes/)。


本模板也附带了一些shortcode如

rawhtml
```
{{< rawhtml >}}
<div>I'm just bored.</div>
{{< /rawhtml >}}
```

video
```
{{< video src="video.mp4" loop=true controls=false autoplay=true title="Cool video for you!" >}}
```

本项目额外实现的shortcode

button用于风格化的按钮
```
{{< button href="https://github.com/QwenLM/Qwen2" label="GITHUB" external=true >}}
```

example用于样例展示
```
{{< example data="1.json" hide=false next=false >}}
```
- data为样例内容同级目录下放置样例的JSON格式文件，格式如下
    ```json
    {
        "title": "Example",
        "messages": [
            {
                "role": "User",
                "content": "Hi!"
            },
            {
                "role": "Qwen",
                "content": "Hi!"
            }
        ]
    }
    ```
    content为hugo支持的markdown语法，可包含shortcode。如需要支持latex渲染，请避免使用`\(...\)`或`\[\]`，因为它们在常用的markdown实现中是对应字符的转义。请使用`$...$`或`$$...$$$`，或者对`\`也进行转义即`\\(...\\)`或`\\[...\\]`。
- hide表示该条目是否初始时隐藏
- next表示是否显示next按钮

其中hide, next用于配合实现样例库效果（需额外设置front matter中gallery为true）
```
{{< fullwidth class="example-container" >}}
{{< example data="1.json" hide=false next=true >}}
{{< example data="2.json" hide=true next=true >}}
{{< example data="3.json" hide=true next=true >}}
{{< example data="4.json" hide=true next=true >}}
{{< /fullwidth >}}
```
- 右击回到顶部按钮，可展开所有样例。

更多请见`themes\PaperMod\layouts\shortcodes`和`layouts\shortcodes`。

### footnote

reference等请使用markdown语法中的footnote编写。
```
This is a footnote[^short].


[^short]: somewhere in content
```
其中`short`可以为任意id，编译后会自动按文中出现顺序编号，文中会包含到文后footnote的链接，footnote后会有到文中引用处的回链。

## 网站配置

见文件[`config.yml`](config.yml)。

发布时注意修改`baseURL`项。

## 自动部署

通过github action可以自动将原代码编译并部署到github pages中。本存储库中已有相关配置，请注意其默认从`main`分支中读取，发布至`gh-pages`分支中。

