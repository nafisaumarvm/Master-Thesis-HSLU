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
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  flex: 1;
`

const LogoCard = styled.div<{ focused: boolean }>`
  background: #15161a;
  border-radius: 16px;
  padding: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 140px;
  border: 2px solid ${({ focused }) => (focused ? '#22d3ee' : 'transparent')};
  box-shadow: ${({ focused }) => (focused ? '0 0 0 4px #22d3ee66' : 'none')};
  font-size: 1.6rem;
`

const Footer = styled.div`
  opacity: 0.8;
`

type Item = { name: string; url: string }

const items: Item[] = [
  { name: 'Netflix', url: 'https://netflix.com' },
  { name: 'Prime Video', url: 'https://primevideo.com' },
  { name: 'YouTube', url: 'https://youtube.com/tv' },
  { name: 'Disney+', url: 'https://disneyplus.com' },
  { name: 'Hulu', url: 'https://hulu.com' },
  { name: 'HBO Max', url: 'https://www.max.com' },
  { name: 'Apple TV+', url: 'https://tv.apple.com' },
  { name: 'Paramount+', url: 'https://www.paramountplus.com' },
]

export function StreamingSection({ onBack }: { onBack: () => void }) {
  const [index, setIndex] = useState(0)
  const columns = 4

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') setIndex((i) => Math.max(0, i - 1))
      if (e.key === 'ArrowRight') setIndex((i) => Math.min(items.length - 1, i + 1))
      if (e.key === 'ArrowUp') setIndex((i) => Math.max(0, i - columns))
      if (e.key === 'ArrowDown') setIndex((i) => Math.min(items.length - 1, i + columns))
      if (e.key === 'Backspace' || e.key === 'Escape') onBack()
      if (e.key === 'Enter') {
        const sel = items[index]
        // For prototype: attempt to open link in same tab
        window.location.href = sel.url
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [index, onBack])

  return (
    <Container>
      <Heading>Streaming Services</Heading>
      <Grid>
        {items.map((it, i) => (
          <LogoCard key={it.name} focused={i === index} className={`focusable ${i === index ? 'focused' : ''}`}>
            {it.name}
          </LogoCard>
        ))}
      </Grid>
      <Footer>Use arrows to navigate, Enter to open, Back/Escape to return</Footer>
    </Container>
  )
}

export default StreamingSection

