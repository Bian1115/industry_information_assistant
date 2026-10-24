import IconHistory from '@/assets/layout/history.svg'
import IconKnowledge from '@/assets/layout/knowledge.svg'
import IconNewChat from '@/assets/layout/newchat.svg'
import { useState } from 'react'
import { useLocation } from 'react-router-dom'
import { NavItem } from './nav-item'
import { SessionDrawer } from '@/components/session-drawer'
import './nav.scss'

export function Nav() {
  const { pathname } = useLocation()
  const [open, setOpen] = useState(false)
  return <>
    <div className="industry-selector"><div className="industry-selector__trigger"><span className="industry-selector__label">AI 深度研究</span></div></div>
    <div className="base-layout-nav">
      <NavItem label="新建研究" icon={IconNewChat} href="/chat" active={pathname === '/chat' || pathname === '/'} />
      <NavItem label="研究与对话历史" icon={IconHistory} href="#" onClick={() => setOpen(true)} />
      <NavItem label="资料知识库" icon={IconKnowledge} href="/knowledge" active={pathname === '/knowledge'} />
    </div>
    <SessionDrawer open={open} onClose={() => setOpen(false)} />
  </>
}
