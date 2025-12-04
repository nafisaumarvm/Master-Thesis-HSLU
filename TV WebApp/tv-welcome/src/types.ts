export type GuestData = {
  name: string
  checkInDate: string
  interests: string[]
}

export type AppRoute =
  | { screen: 'ad' }
  | { screen: 'welcome' }
  | { screen: 'streaming' }
  | { screen: 'info' }

