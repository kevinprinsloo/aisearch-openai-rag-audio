declare module 'linkify-it' {
    interface LinkifyIt {
      (options?: any): LinkifyIt;
      add(schema: string, rule: RegExp): LinkifyIt;
      match(text: string): Array<{ index: number; lastIndex: number; raw: string; schema: string; text: string; url: string }>;
      normalize(raw: string): string;
      pretest(text: string): boolean;
      test(text: string): boolean;
      tlds(list: string | string[], keepOld?: boolean): LinkifyIt;
    }
  
    const linkify: LinkifyIt;
    export default linkify;
  }