// Procedural place names from syllables, stable per seed.
import { hash2i } from './spec.js'

const HEAD = ['ar', 'bel', 'cor', 'dun', 'el', 'fen', 'gal', 'hal', 'ist', 'jor', 'kel', 'lor', 'mar', 'nor', 'or', 'pel',
  'quin', 'ros', 'sal', 'tor', 'ul', 'val', 'wen', 'yar', 'zan', 'bra', 'cal', 'dor', 'esk', 'fal', 'gri', 'hav', 'kas',
  'lin', 'mir', 'nev', 'os', 'pra', 'riv', 'sten', 'tal', 'ver', 'aln', 'bry', 'cas', 'del', 'eth', 'lum']
const MID = ['a', 'e', 'i', 'o', 'u', 'ia', 'ae', 'en', 'ar', '', '', '']
const TAIL = ['burg', 'ford', 'holm', 'mouth', 'stad', 'ville', 'port', 'bridge', 'dale', 'haven', 'wick', 'by', 'sund',
  'mere', 'field', 'ton', 'gard', 'polis', 'ia', 'ora', 'enza', 'ino', 'ak', 'ec', 'is', 'um']

export function placeName(seed) {
  const pick = (list, k) => list[Math.floor(hash2i(seed, k, 101) * list.length)]
  const s = pick(HEAD, 1) + pick(MID, 2) + pick(TAIL, 3)
  return s[0].toUpperCase() + s.slice(1)
}
