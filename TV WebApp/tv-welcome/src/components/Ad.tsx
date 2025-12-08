import { useEffect, useState } from 'react'
import styled from 'styled-components'

const Container = styled.div`
  width: 100vw;
  height: 100vh;
  position: relative;
  background: #0a0a0a;
  overflow: hidden;
`

const BackgroundVideo = styled.video`
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0.9;
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

const MainText = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 3;
  text-align: center;
  color: white;
  font-size: 3rem;
  font-weight: bold;
  text-shadow: 0 4px 24px rgba(0,0,0,0.8);
  line-height: 1.2;
  max-width: 80%;
`

const QRWrapper = styled.div`
  position: absolute;
  top: 80px;
  right: 80px;
  width: 200px;
  height: 200px;
  background: white;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3;
  padding: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
`

const QRImage = styled.img`
  width: 100%;
  height: 100%;
  object-fit: contain;
`

const SkipButton = styled.button`
  position: absolute;
  bottom: 32px;
  right: 32px;
  z-index: 4;
  background: rgba(0,0,0,0.7);
  color: white;
  border: 2px solid white;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 1.2rem;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255,255,255,0.1);
  }
  
  &:focus {
    outline: 3px solid #22d3ee;
    outline-offset: 4px;
  }
`

const Timer = styled.div`
  position: absolute;
  bottom: 32px;
  right: 200px;
  z-index: 3;
  color: white;
  font-size: 1.4rem;
  background: rgba(0,0,0,0.7);
  padding: 8px 16px;
  border-radius: 8px;
`

const SponsoredLabel = styled.div`
  position: absolute;
  top: 24px;
  left: 24px;
  z-index: 3;
  color: white;
  font-size: 1rem;
  background: rgba(0,0,0,0.7);
  padding: 8px 16px;
  border-radius: 8px;
  font-weight: 500;
  letter-spacing: 0.5px;
`

type Props = {
  onComplete: () => void
}

export function UberAd({ onComplete }: Props) {
  const [timeLeft, setTimeLeft] = useState(0)
  const [canSkip, setCanSkip] = useState(false)

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          return 0
        }
        return prev - 1
      })
    }, 1000)

    // Enable skip after 30 seconds
    const skipTimer = setTimeout(() => {
      setCanSkip(true)
    }, 30000)

    return () => {
      clearInterval(timer)
      clearTimeout(skipTimer)
    }
  }, [onComplete])

  const handleVideoLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
    const video = e.currentTarget
    setTimeLeft(Math.floor(video.duration))
  }

  const handleVideoEnded = () => {
    onComplete()
  }

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter' && canSkip) {
        onComplete()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [canSkip, onComplete])

  return (
    <Container>
      <BackgroundVideo 
        src="/Verkehrshaus der Schweiz – Swiss Museum of Transport.mp4" 
        autoPlay
        muted
        onLoadedMetadata={handleVideoLoadedMetadata}
        onEnded={handleVideoEnded}
      />
      
      <SponsoredLabel>Sponsored Content</SponsoredLabel>
      
      <LeftRail>
        <span>⌕</span>
        <span>▦</span>
        <span>▶</span>
        <span>♥</span>
        <span>👤</span>
      </LeftRail>
      
      <MainText>
        Come and Discover the Swiss Museum of Transport,<br />
        just 5 minutes walk away
      </MainText>
      
      <QRWrapper>
        <QRImage 
          src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://www.verkehrshaus.ch/en/"
          alt="Visit Verkehrshaus Website"
        />
      </QRWrapper>
      
      <Timer>{timeLeft}s</Timer>
      
      {canSkip && (
        <SkipButton 
          className="focusable focused"
          onClick={onComplete}
        >
          Skip Ad
        </SkipButton>
      )}
    </Container>
  )
}

export default UberAd
export { UberAd as Ad }