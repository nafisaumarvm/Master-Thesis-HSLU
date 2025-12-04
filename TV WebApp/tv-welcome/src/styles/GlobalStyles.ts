import { createGlobalStyle } from 'styled-components'

export const GlobalStyles = createGlobalStyle`
  :root {
    color-scheme: dark;
  }

  * {
    box-sizing: border-box;
  }

  html, body, #root {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: #0b0b0c;
  }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
      Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    color: #f3f4f6;
    line-height: 1.4;
  }

  /* TV-friendly base font sizes and focus rings */
  :focus {
    outline: 3px solid #22d3ee;
    outline-offset: 4px;
  }

  /* Hide default cursor for TV */
  body {
    cursor: none;
  }

  /* Utility classes */
  .screen {
    width: 100vw;
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px;
  }

  .focusable {
    transition: transform 120ms ease, box-shadow 120ms ease;
  }

  .focused {
    transform: scale(1.04);
    box-shadow: 0 0 0 4px #22d3ee66;
  }
`

export default GlobalStyles