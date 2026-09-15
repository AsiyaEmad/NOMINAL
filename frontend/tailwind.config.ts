import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#070A10',
        panel: '#0D111B',
        line: '#202737',
        signal: '#9CF7C8',
        violet: '#A99BFF',
        amber: '#FFCA76',
      },
      boxShadow: {
        glow: '0 0 40px rgba(156, 247, 200, 0.08)',
      },
    },
  },
  plugins: [],
} satisfies Config
