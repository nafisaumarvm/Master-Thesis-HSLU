import { useEffect, useRef, useState } from 'react'
import styled from 'styled-components'
import type { GuestData } from '../types'

type AdContent = {
  title: string
  description: string
  imageUrl?: string
  videoUrl?: string
  ctaText?: string
}

const Container = styled.div`
  width: 100vw;
  height: 100vh;
  position: relative;
  background: #101114;
  overflow: hidden;
`

const Content = styled.div`
  position: absolute;
  top: 12vh;
  left: 8vw;
  z-index: 2;
  text-align: left;
  max-width: 50vw;
`

const Title = styled.h1`
  font-size: 5rem;
  margin: 0 0 16px 0;
  text-shadow: 0 4px 24px rgba(0,0,0,0.7);
`

const Description = styled.p`
  font-size: 2rem;
  margin: 0 0 32px 0;
  color: #e5e7eb;
  text-shadow: 0 2px 16px rgba(0,0,0,0.7);
`

const CTA = styled.button`
  padding: 22px 36px;
  font-size: 1.8rem;
  border-radius: 18px;
  background: #22d3ee;
  color: #071317;
  border: none;
`

const BackgroundImage = styled.img`
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.6;
  filter: saturate(1.05);
`

const Video = styled.video`
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.6;
  filter: saturate(1.1);
`

const LeftRail = styled.div`
  position: absolute;
  left: 24px;
  top: 0;
  bottom: 0;
  width: 64px;
  display: flex;
  flex-direction: column;
  gap: 24px;
  align-items: center;
  justify-content: center;
  z-index: 3;
  color: #a3a3a3;
  opacity: 0.75;
`

const Sponsor = styled.div`
  position: absolute;
  top: 32px;
  left: 32px;
  z-index: 3;
  color: #e5e7eb;
  opacity: 0.9;
  font-weight: 600;
`

const QRWrapper = styled.div`
  position: absolute;
  right: 4vw;
  bottom: 8vh;
  width: 180px;
  height: 180px;
  background: rgba(0,0,0,0.6);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3;
`

export function AdPage({ guest, onContinue }: { guest: GuestData; onContinue: () => void }) {
  const [ad, setAd] = useState<AdContent | null>(null)
  const buttonRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    console.log('[ad] mount for', guest.name)
    // Mock dynamic ad load based on interests
    const firstInterest = guest.interests[0] ?? 'travel'
    const mock: AdContent = {
      title: `Hi ${guest.name}, unlock special ${firstInterest} perks!`,
      description: `Staying from ${guest.checkInDate}? Enjoy exclusive ${firstInterest} offers during your stay.`,
      imageUrl: 'https://images.unsplash.com/photo-1502920917128-1aa500764cbd?q=80&w=1920&auto=format&fit=crop',
      ctaText: 'Continue to Welcome Page',
    }
    setAd(mock)
  }, [guest])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      // Samsung Tizen browsers map remote keys to standard key events
      if (e.key === 'Enter') {
        onContinue()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onContinue])

  useEffect(() => {
    buttonRef.current?.focus()
  }, [ad])

  // Render a minimal shell immediately to avoid a blank screen while ad loads
  if (!ad) {
    return (
      <Container className="screen">
        <Sponsor>Sponsored Content</Sponsor>
        <Content>
          <Title>Loading…</Title>
          <Description>Please wait</Description>
        </Content>
      </Container>
    )
  }

  return (
    <Container className="screen">
      {ad.videoUrl ? <Video src={ad.videoUrl} autoPlay muted loop /> : ad.imageUrl ? <BackgroundImage src={ad.imageUrl} alt="ad" /> : null}
      <Sponsor>Sponsored Content</Sponsor>
      <LeftRail>
        <span>⌕</span>
        <span>▦</span>
        <span>▶</span>
        <span>♥</span>
        <span>👤</span>
      </LeftRail>
      <Content>
        <Title>{ad.title}</Title>
        <Description>{ad.description}</Description>
        <CTA ref={buttonRef} className="focusable focused" onClick={onContinue}>
          {ad.ctaText ?? 'Continue'}
        </CTA>
      </Content>
      <QRWrapper>
        <div style={{ 
          width: '150px', 
          height: '150px', 
          background: '#000', 
          color: '#fff', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          fontSize: '1.2rem',
          textAlign: 'center'
        }}>
          QR Code<br />Placeholder
        </div>
      </QRWrapper>
    </Container>
  )
}

export default AdPage

