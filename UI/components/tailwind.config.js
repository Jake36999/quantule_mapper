/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gold: {
          500: '#c5a059', // V13 Gold branding
        }
      }
    },
  },
  plugins: [],
}
