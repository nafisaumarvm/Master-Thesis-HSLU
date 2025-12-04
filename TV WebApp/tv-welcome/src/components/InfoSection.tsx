import { useEffect, useState } from 'react'
import styled from 'styled-components'

const Container = styled.div`
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  padding: 48px;
  gap: 24px;
`

const Heading = styled.h2`
  font-size: 2.8rem;
  margin: 0 0 8px 0;
`

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  flex: 1;
`

const Card = styled.div<{ focused: boolean }>`
  background: #15161a;
  border-radius: 16px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 160px;
  border: 2px solid ${({ focused }) => (focused ? '#22d3ee' : 'transparent')};
  box-shadow: ${({ focused }) => (focused ? '0 0 0 4px #22d3ee66' : 'none')};
  font-size: 1.4rem;
`

type Item = { title: string; lines: string[] }

const items: Item[] = [
  { title: 'Wi‑Fi', lines: ['Network: StayGuest', 'Password: welcome2025'] },
  { title: 'Room Service', lines: ['Dial 0 from room phone', 'Service hours: 6:00–23:00'] },
  { title: 'Front Desk', lines: ['WhatsApp: +1 555 123 4567', 'Email: hello@stay.example'] },
  { title: 'Offers', lines: ['Spa −10% w/ code TVSPA', 'Late checkout on request'] },
  { title: 'Mobile App', lines: ['Scan QR to download', 'iOS & Android supported'] },
  { title: 'Emergency', lines: ['Dial 112', 'Stairwell next to elevator'] },
]

export function InfoSection({ onBack }: { onBack: () => void }) {
  const [index, setIndex] = useState(0)
  const columns = 3

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(0, i - 1))
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(items.length - 1, i + 1))
      if (e.key === 'ArrowUp') setIndex((i) => Math.max(0, i - columns))
      if (e.key === 'ArrowDown') setIndex((i) => Math.min(items.length - 1, i + columns))
      if (e.key === 'Backspace' || e.key === 'Escape') onBack()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onBack])

  return (
    <Container>
      <Heading>Hotel / Airbnb Info</Heading>
      <Grid>
        {items.map((it, i) => (
          <Card key={it.title} focused={i === index} className={`focusable ${i === index ? 'focused' : ''}`}>
            <strong style={{ fontSize: '1.6rem' }}>{it.title}</strong>
            {it.lines.map((line) => (
              <span key={line}>{line}</span>
            ))}
          </Card>
        ))}
      </Grid>
      <div style={{ opacity: 0.8 }}>Use arrows to navigate, Back/Escape to return</div>
    </Container>
  )
}

export default InfoSection

