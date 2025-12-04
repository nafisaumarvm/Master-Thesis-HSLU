import { useMemo, useState } from 'react'
import { Ad } from './components/Ad'
import { WelcomePage } from './components/WelcomePage'
import { StreamingSection } from './components/StreamingSection'
import { InfoSection } from './components/InfoSection'
import type { AppRoute, GuestData } from './types'

function App() {
  const [route, setRoute] = useState<AppRoute>({ screen: 'ad' })
  console.log('[app] route', route)

  const guest: GuestData = useMemo(
    () => ({
      name: 'Peter',
      checkInDate: '2025-10-01',
      interests: ['spa', 'dining', 'city-tours'],
    }),
    []
  )

  if (route.screen === 'ad') {
    return <Ad onComplete={() => setRoute({ screen: 'welcome' })} />
  }

  if (route.screen === 'welcome') {
    return (
      <WelcomePage
        guest={guest}
        onNavigate={(dest) => {
          if (dest === 'streaming') setRoute({ screen: 'streaming' })
          if (dest === 'info') setRoute({ screen: 'info' })
          if (dest === 'exit') setRoute({ screen: 'ad' })
        }}
      />
    )
  }

  if (route.screen === 'streaming') {
    return <StreamingSection onBack={() => setRoute({ screen: 'welcome' })} />
  }

  if (route.screen === 'info') {
    return <InfoSection onBack={() => setRoute({ screen: 'welcome' })} />
  }

  return null
}

export default App