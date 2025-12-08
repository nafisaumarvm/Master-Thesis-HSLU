import { useEffect, useState } from 'react'
import styled from 'styled-components'
import type { GuestData } from '../types'

const Container = styled.div`
  width: 100vw;
  height: 100vh;
  position: relative;
  display: flex;
  flex-direction: column;
  padding: 48px;
  gap: 40px;
  background: #0b0b0c;
  overflow: hidden;
`

const Hero = styled.img`
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 65vw;
  object-fit: cover;
  opacity: 0.9;
`

const Heading = styled.h1`
  font-size: 4.8rem;
  margin: 0;
  text-shadow: 0 4px 24px rgba(0,0,0,0.7);
`

const Sections = styled.div`
  display: grid;
  grid-template-columns: repeat(5, minmax(280px, 1fr));
  gap: 24px;
  position: absolute;
  left: 80px;
  right: 80px;
  bottom: 60px;
`

const Card = styled.div<{ focused: boolean }>`
  border-radius: 20px;
  padding: 0;
  background: #15161a;
  min-height: 200px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 2rem;
  border: 2px solid ${({ focused }) => (focused ? '#22d3ee' : 'transparent')};
  box-shadow: ${({ focused }) => (focused ? '0 0 0 4px #22d3ee66' : 'none')};
`

const Thumb = styled.div`
  height: 160px;
  background: #0f172a;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #cbd5e1;
`

const Label = styled.div`
  padding: 18px 20px;
`

type Props = {
  guest: GuestData
  onNavigate: (dest: 'streaming' | 'info' | 'exit') => void
}

const items: Array<{ key: 'streaming' | 'info' | 'exit'; label: string }> = [
  { key: 'streaming', label: 'Television' },
  { key: 'info', label: 'Dining' },
  { key: 'info', label: 'Guest Services' },
  { key: 'info', label: 'Information' },
  { key: 'streaming', label: 'Streaming' },
]

export function WelcomePage({ guest, onNavigate }: Props) {
  const [index, setIndex] = useState(0)

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(0, i - 1))
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(items.length - 1, i + 1))
      if (e.key === 'Enter') {
        const sel = items[index]
        if (sel.key === 'exit') {
          onNavigate('exit')
        } else if (sel.key === 'streaming') {
          onNavigate('streaming')
        } else if (sel.key === 'info') {
          onNavigate('info')
        }
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [index, onNavigate])

  return (
    <Container>
      <Hero src="https://images.unsplash.com/photo-1505761671935-60b3a7427bad?q=80&w=2400&auto=format&fit=crop" alt="city" />
      <Heading>Welcome, {guest.name}!</Heading>
      <div style={{ color: '#d1d5db', maxWidth: '50vw' }}>Enjoy your stay with us. Explore services and amenities right from your TV.</div>
      <Sections>
        {items.map((it, i) => (
          <Card key={`${it.label}-${i}`} focused={i === index} className={`focusable ${i === index ? 'focused' : ''}`}>
            <Thumb>{it.label}</Thumb>
            <Label>{it.label}</Label>
          </Card>
        ))}
      </Sections>
    </Container>
  )
}

export default WelcomePage

