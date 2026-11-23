import { createContext, useContext, type ReactNode } from 'react';
import { TOKENS, type ChartTokens } from './theme';

// The page's only shared state is the chart colours; the selected setting lives in the address bar (StudyTab).
interface Ctx { tokens: ChartTokens }
const VALUE: Ctx = { tokens: TOKENS };
const ViewContext = createContext<Ctx>(VALUE);

export const useView = (): Ctx => useContext(ViewContext);

export function ViewProvider({ children }: { children: ReactNode }) {
  return <ViewContext.Provider value={VALUE}>{children}</ViewContext.Provider>;
}
