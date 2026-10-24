/**
 * Copyright © 2026 深圳市深维智见教育科技有限公司 版权所有
 * 未经授权，禁止转售或仿制。
 */

import classNames from 'classnames'
import { Marked, Renderer, TokenizerAndRendererExtension } from 'marked'
import { useMemo } from 'react'
import './index.scss'

// 文献与模型生成的 Markdown 均来自外部内容，只保留展示所需的元素和属性。
export function sanitizeMarkdownHtml(html: string): string {
  const document = new DOMParser().parseFromString(html, 'text/html')
  const allowed = new Set(['P', 'BR', 'HR', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'STRONG', 'EM', 'DEL', 'S', 'BLOCKQUOTE', 'PRE', 'CODE', 'UL', 'OL', 'LI', 'TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'A', 'IMG', 'DIV', 'SPAN', 'SUP', 'SUB'])
  for (const element of Array.from(document.body.querySelectorAll('*'))) {
    if (!allowed.has(element.tagName)) { element.remove(); continue }
    for (const attribute of Array.from(element.attributes)) {
      if (!['href', 'src', 'alt', 'title', 'class', 'start', 'colspan', 'rowspan'].includes(attribute.name)) element.removeAttribute(attribute.name)
    }
    if (element.hasAttribute('href')) {
      const href = element.getAttribute('href') || ''
      if (!/^(https?:\/\/|#[A-Za-z0-9_-])/i.test(href)) element.removeAttribute('href')
      else { element.setAttribute('target', '_blank'); element.setAttribute('rel', 'noopener noreferrer') }
    }
    if (element.hasAttribute('src') && !/^(https?:\/\/|data:image\/(png|jpeg|gif|webp);base64,)/i.test(element.getAttribute('src') || '')) element.removeAttribute('src')
  }
  return document.body.innerHTML
}

export default function Markdown(props: {
  className?: string
  value?: string
  extensions?: TokenizerAndRendererExtension[]
}) {
  const { value, extensions, className, ...otherProps } = props

  const html = useMemo(() => {
    const renderer = new Renderer()

    // 自定义图片渲染：只渲染有效的图片 URL，隐藏无效的图片引用
    renderer.image = ({ href, title, text }: { href: string; title: string | null; text: string }) => {
      // 只渲染 base64 data URL 或有效的 http(s) URL
      if (href && (href.startsWith('data:image/') || href.startsWith('http://') || href.startsWith('https://'))) {
        const titleAttr = title ? ` title="${title}"` : ''
        return `<img src="${href}" alt="${text || ''}"${titleAttr} class="markdown-image" loading="lazy" />`
      }
      // 无效的图片 URL，直接隐藏（实际图表在"可视化图表"tab中显示）
      return ''
    }

    const marked = new Marked({
      extensions: props.extensions,
    })
    const html = marked.parse(props.value ?? '', {
      gfm: true,
      renderer,
    })

    return sanitizeMarkdownHtml(String(html))
  }, [value, extensions])

  return (
    <div
      className={classNames('com-markdown', className)}
      {...otherProps}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
